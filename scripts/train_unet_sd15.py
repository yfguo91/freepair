#!/usr/bin/env python3
"""
Stage 2: fine-tune the SD1.5 UNet with **full-parameter** AdamW on the fused
latent z0 = E(F(x_vis, x_ir)) * s (Eq. 6); the dual VAE stays frozen.

Objective (Eq. 7): epsilon-MSE + lambda_P * L_P, where L_P is the infrared
statistical-moment prior -- the epsilon prediction is converted to an x0
prediction, decoded through the frozen IR branch, and its per-sample
(mean, std) are matched to the ground-truth IR (Eq. 8). `aux_timestep_max`
optionally restricts L_P to timesteps t <= tau.

CLIP caption conditioning: captions come from the dataset captions JSON.

  python scripts/train_unet_sd15.py --config configs/train_unet_m3fd_v2.yaml
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sd_dual.data_pair import M3FDPairDatasetSD
from sd_dual.diffusion_aux_loss import auxiliary_decode_losses, fused_latents_for_training
from sd_dual.dual_vae import DualBranchSDVAE
from sd_dual.sd15_ddim_sample import ddim_sample_scaled_latents

try:
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError as e:  # pragma: no cover
    raise ImportError("Install diffusers + transformers: pip install -r requirements.txt") from e

# train_loss_by_epoch.csv: fixed column names so DictWriter stays consistent across epochs
EPOCH_LOG_CSV_FIELDS = (
    "epoch",
    "global_step",
    "avg_noise_mse",
    "avg_ir_stat",
    "avg_ir_stat_weighted",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.inference_mode()
def save_train_preview(
    *,
    unet: UNet2DConditionModel,
    dual: DualBranchSDVAE,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    sd_id: str,
    local_files_only: bool,
    device: torch.device,
    out_path: Path,
    prompt: str,
    num_samples: int,
    num_inference_steps: int,
    guidance_scale: float,
    img_size: int,
    seed: int,
) -> None:
    """DDIM sampling -> dual-decoder VAE decode; saves a stacked VIS|IR grid."""
    dec_dtype = dual.decoder_vis.conv_in.weight.dtype
    n = max(1, int(num_samples))
    was_training = unet.training
    unet.eval()
    try:
        latents = ddim_sample_scaled_latents(
            unet=unet,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            sd_id=sd_id,
            local_files_only=local_files_only,
            device=device,
            prompts=[prompt] * n,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            img_size=img_size,
            seed=seed,
        )
    finally:
        if was_training:
            unet.train()

    z = latents.to(dtype=dec_dtype) / dual.scaling_factor
    vis = dual.decode_vis(z)
    ir = dual.decode_ir(z)
    if ir.shape[1] == 1:
        ir = ir.repeat(1, 3, 1, 1)
    vis_n = (vis.clamp(-1, 1) + 1) * 0.5
    ir_n = (ir.clamp(-1, 1) + 1) * 0.5
    rows = []
    for i in range(n):
        rows.append(torch.cat([vis_n[i : i + 1], ir_n[i : i + 1]], dim=3))
    grid = torch.cat(rows, dim=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path), nrow=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train_unet_m3fd_v2.yaml")
    args = ap.parse_args()
    cfg = load_yaml(ROOT / args.config)

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    sd_id = cfg["paths"]["sd_id"]
    lfo = bool(cfg["paths"].get("local_files_only", False))

    # Frozen dual VAE from Stage 1
    dual_ckpt = Path(cfg["paths"]["dual_vae_ckpt"])
    ck = torch.load(dual_ckpt, map_location=device, weights_only=False)
    dual = DualBranchSDVAE(
        pretrained_model=sd_id,
        subfolder="vae",
        local_files_only=lfo,
        ir_out_channels=int(cfg["model"].get("ir_out_channels", 1)),
    ).to(device)
    info = dual.load_state_dict(ck["state_dict"], strict=False)
    if info.unexpected_keys:
        print(f"[warn] unexpected keys in dual ckpt: {info.unexpected_keys}")
    if info.missing_keys:
        print(f"[warn] missing keys when loading dual ckpt: {info.missing_keys}")
    dual.eval()
    for p in dual.parameters():
        p.requires_grad = False

    tokenizer = CLIPTokenizer.from_pretrained(
        sd_id, subfolder="tokenizer", local_files_only=lfo
    )
    text_encoder = CLIPTextModel.from_pretrained(
        sd_id, subfolder="text_encoder", local_files_only=lfo
    ).to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    unet = UNet2DConditionModel.from_pretrained(
        sd_id, subfolder="unet", local_files_only=lfo
    ).to(device)
    for p in unet.parameters():
        p.requires_grad = True
    unet.train()
    print(
        f"[diffusion] full fine-tune | trainable params: "
        f"{sum(p.numel() for p in unet.parameters() if p.requires_grad):,}"
    )
    opt = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]),
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        sd_id, subfolder="scheduler", local_files_only=lfo
    )
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

    data_cfg = cfg["data"]
    cj = data_cfg.get("captions_json")
    captions_json = Path(cj).resolve() if cj else None
    ds = M3FDPairDatasetSD(
        root=data_cfg["root"],
        img_size=int(data_cfg["img_size"]),
        max_samples=data_cfg.get("max_samples"),
        default_caption=str(data_cfg.get("default_caption", "")),
        captions_json=captions_json,
        ir_channels=int(data_cfg.get("ir_channels", 1)),
        vis_dir_name=str(data_cfg.get("vis_dir_name", "vi")),
        ir_dir_name=str(data_cfg.get("ir_dir_name", "ir")),
        pair_mode=str(data_cfg.get("pair_mode", "same_name")),
        id_regex_vis=str(data_cfg.get("id_regex_vis", r"FLIR_(\d+)")),
        id_regex_ir=str(data_cfg.get("id_regex_ir", r"FLIR_(\d+)")),
    )
    print(
        "[data] "
        f"root={data_cfg['root']} | vis_dir={data_cfg.get('vis_dir_name', 'vi')} | "
        f"ir_dir={data_cfg.get('ir_dir_name', 'ir')} | pair_mode={data_cfg.get('pair_mode', 'same_name')} | "
        f"samples={len(ds)}"
    )
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )

    use_amp = bool(cfg["train"].get("use_amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    max_steps = int(cfg["train"]["max_train_steps"])
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])

    tcfg = cfg["train"]
    aux_lambda_ir_stat = float(tcfg.get("aux_lambda_ir_stat", 0.1))
    aux_z_clip = float(tcfg.get("aux_z_clip", 10.0))
    _atm = tcfg.get("aux_timestep_max", None)
    aux_timestep_max = int(_atm) if _atm is not None else None
    aux_active = aux_lambda_ir_stat > 0.0
    print(
        f"[diffusion] IR statistical-moment prior L_P: {'ON' if aux_active else 'OFF'} | "
        f"lambda_P={aux_lambda_ir_stat} | aux_timestep_max={aux_timestep_max} | z_clip={aux_z_clip}"
    )

    out_root = Path(cfg["output"]["root"])
    out_root.mkdir(parents=True, exist_ok=True)
    preview_dir = out_root / "train_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_root / "train_loss_by_epoch.csv"

    preview_every_epochs = int(cfg["train"].get("preview_every_epochs", 1))
    preview_every_steps = int(cfg["train"].get("preview_every_steps", 0))
    preview_inference_steps = int(cfg["train"].get("preview_inference_steps", 30))
    preview_guidance_scale = float(cfg["train"].get("preview_guidance_scale", 7.5))
    preview_num_samples = int(cfg["train"].get("preview_num_samples", 2))
    preview_seed_base = int(cfg["train"].get("preview_seed", int(cfg["train"]["seed"])))
    img_size = int(data_cfg["img_size"])

    step = 0
    epoch_idx = 0
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(total=max_steps, desc="diffusion")
    while step < max_steps:
        ep_noise_mse = 0.0
        ep_ir_stat = 0.0
        ep_ir_stat_w = 0.0
        ep_batches = 0
        for x_vis, x_ir, names, caps in dl:
            if step >= max_steps:
                break
            x_vis = x_vis.to(device, non_blocking=True)
            x_ir = x_ir.to(device, non_blocking=True)
            bsz = x_vis.shape[0]

            with torch.cuda.amp.autocast(enabled=use_amp):
                # clean fused latent target (Eq. 6)
                with torch.no_grad():
                    latents = fused_latents_for_training(dual, x_vis, x_ir)

                tok = tokenizer(
                    list(caps),
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = tok.input_ids.to(device)
                with torch.no_grad():
                    enc_out = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=device,
                    dtype=torch.long,
                )
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                pred = unet(
                    noisy,
                    timesteps,
                    encoder_hidden_states=enc_out,
                ).sample
                loss_noise = F.mse_loss(pred, noise)

                l_ir_stat_term = torch.zeros((), device=device, dtype=torch.float32)
                aux_d = None
                if aux_active:
                    with torch.cuda.amp.autocast(enabled=False):
                        aux_d = auxiliary_decode_losses(
                            dual=dual,
                            noisy_latents=noisy.float(),
                            noise_pred=pred.float(),
                            timesteps=timesteps,
                            alphas_cumprod=alphas_cumprod,
                            x_ir_gt=x_ir.float(),
                            z_clip=aux_z_clip,
                            aux_timestep_max=aux_timestep_max,
                        )
                        l_ir_stat_term = aux_lambda_ir_stat * aux_d["ir_stat"]

                loss = loss_noise + l_ir_stat_term

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            ep_noise_mse += float(loss_noise.detach().item())
            if aux_active and aux_d is not None:
                ep_ir_stat += float(aux_d["ir_stat"].detach().item())
                ep_ir_stat_w += float(l_ir_stat_term.detach().item())
            ep_batches += 1
            step += 1
            pbar.update(1)
            if step % log_every == 0:
                pf: Dict[str, float] = {"mse": float(loss_noise.item()), "tot": float(loss.item())}
                if aux_active and aux_d is not None:
                    pf["lp"] = float(aux_d["ir_stat"].item())
                pbar.set_postfix(**{k: round(v, 5) for k, v in pf.items()})

            if save_every > 0 and step % save_every == 0:
                torch.save(
                    {"step": step, "unet": unet.state_dict(), "config": cfg},
                    out_root / f"unet_step_{step}.pt",
                )

            if preview_every_steps > 0 and step > 0 and step % preview_every_steps == 0:
                prev_path = preview_dir / f"step_{step:07d}.png"
                was_tr = unet.training
                try:
                    unet.eval()
                    save_train_preview(
                        unet=unet,
                        dual=dual,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        sd_id=sd_id,
                        local_files_only=lfo,
                        device=device,
                        out_path=prev_path,
                        prompt="",
                        num_samples=preview_num_samples,
                        num_inference_steps=preview_inference_steps,
                        guidance_scale=preview_guidance_scale,
                        img_size=img_size,
                        seed=preview_seed_base + step,
                    )
                    print(f"[preview every {preview_every_steps} steps] saved {prev_path}")
                except Exception as e:
                    print(f"[preview step] skipped (reason: {e})")
                finally:
                    if was_tr:
                        unet.train()

        # ---- end of epoch ----
        if ep_batches > 0:
            avg_mse = ep_noise_mse / ep_batches
            row: Dict[str, Any] = {
                "epoch": epoch_idx + 1,
                "global_step": step,
                "avg_noise_mse": round(avg_mse, 8),
                "avg_ir_stat": round(ep_ir_stat / ep_batches, 8) if aux_active else "",
                "avg_ir_stat_weighted": round(ep_ir_stat_w / ep_batches, 8) if aux_active else "",
            }
            extra = (
                f" | L_P: raw={row['avg_ir_stat']:.6f} "
                f"lambda_P*L_P={row['avg_ir_stat_weighted']:.6f}"
                if aux_active
                else ""
            )
            print(
                f"\n========== epoch {epoch_idx + 1} | steps {step}/{max_steps} | "
                f"mean noise_mse(eps) {avg_mse:.6f} ({ep_batches} steps this epoch){extra} =========="
            )
            fe = log_csv.is_file()
            row_csv = {k: row.get(k, "") for k in EPOCH_LOG_CSV_FIELDS}
            with open(log_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(EPOCH_LOG_CSV_FIELDS))
                if not fe:
                    w.writeheader()
                w.writerow(row_csv)

            if preview_every_epochs > 0 and (epoch_idx + 1) % preview_every_epochs == 0:
                prev_path = preview_dir / f"epoch_{epoch_idx + 1:04d}.png"
                was_training = unet.training
                try:
                    unet.eval()
                    save_train_preview(
                        unet=unet,
                        dual=dual,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        sd_id=sd_id,
                        local_files_only=lfo,
                        device=device,
                        out_path=prev_path,
                        prompt="",
                        num_samples=preview_num_samples,
                        num_inference_steps=preview_inference_steps,
                        guidance_scale=preview_guidance_scale,
                        img_size=img_size,
                        seed=preview_seed_base + epoch_idx,
                    )
                    print(f"[preview] saved {prev_path}")
                except Exception as e:
                    print(f"[preview] skipped (reason: {e})")
                finally:
                    if was_training:
                        unet.train()

        epoch_idx += 1
        if step >= max_steps:
            break

    pbar.close()
    torch.save(
        {"step": step, "unet": unet.state_dict(), "config": cfg},
        out_root / "unet_last.pt",
    )
    print(f"Saved to {out_root}")


if __name__ == "__main__":
    main()
