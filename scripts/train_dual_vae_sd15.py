#!/usr/bin/env python3
"""
Stage 1: train the shared-latent dual-stream VAE (Eq. 3-5).

Loss (Eq. 4): L1 reconstruction on both modalities + KL regularizer
+ LPIPS (both modalities) + SSIM (both modalities) + thermal-distribution
matching (first two moments of the IR intensity, IR only).

Run from repo root:
  python scripts/train_dual_vae_sd15.py --config configs/train_vae_m3fd_v2.yaml
"""
from __future__ import annotations

import argparse
import csv
import math
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

LPIPS_IMPORT_ERROR: Exception | None = None
try:
    import lpips as lpips_lib
except Exception as e:  # noqa: BLE001
    lpips_lib = None
    LPIPS_IMPORT_ERROR = e

from sd_dual.data_pair import M3FDPairDatasetSD
from sd_dual.dual_vae import DualBranchSDVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def psnr_approx(a: torch.Tensor, b: torch.Tensor) -> float:
    """[-1,1] tensors, mean over batch; PSNR with peak dynamic range 2."""
    mse = F.mse_loss(a, b, reduction="mean").item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(4.0 / mse)


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def ssim_loss_gray(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Lightweight differentiable SSIM loss on grayscale tensors in [-1, 1].
    Returns 1-SSIM (lower is better).
    """
    if x.shape[1] != 1:
        x = rgb_to_gray(x)
    if y.shape[1] != 1:
        y = rgb_to_gray(y)
    # map to [0,1] for stable constants
    x = (x.clamp(-1, 1) + 1.0) * 0.5
    y = (y.clamp(-1, 1) + 1.0) * 0.5
    k = 3
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    mu_y = F.avg_pool2d(y, kernel_size=k, stride=1, padding=k // 2)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.avg_pool2d(x * x, k, 1, k // 2) - mu_x2
    sigma_y2 = F.avg_pool2d(y * y, k, 1, k // 2) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, k, 1, k // 2) - mu_xy
    ssim_n = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    ssim_d = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim = ssim_n / (ssim_d + 1e-8)
    return (1.0 - ssim).mean()


def thermal_distribution_loss_gray(pred_gray: torch.Tensor, gt_gray: torch.Tensor) -> torch.Tensor:
    """
    Thermal-distribution matching loss L_th (Eq. 5): MSE on the first two
    moments (mean, std) of the IR intensity distribution, on [0,1] gray values.
    pred_gray / gt_gray: [B,1,H,W] in [-1,1]
    """
    p = (pred_gray.clamp(-1, 1) + 1.0) * 0.5
    g = (gt_gray.clamp(-1, 1) + 1.0) * 0.5
    dims = (1, 2, 3)
    mu_p = p.mean(dim=dims)
    mu_g = g.mean(dim=dims)
    var_p = p.var(dim=dims, unbiased=False)
    var_g = g.var(dim=dims, unbiased=False)
    std_p = torch.sqrt(var_p + 1e-8)
    std_g = torch.sqrt(var_g + 1e-8)
    return F.mse_loss(mu_p, mu_g) + F.mse_loss(std_p, std_g)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dtype_from_str(s: str):
    m = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return m.get(s, torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train_vae_m3fd_v2.yaml")
    args = ap.parse_args()
    cfg = load_yaml(ROOT / args.config)

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

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
    preview_n = int(cfg["train"].get("recon_preview_num", 4))
    preview_loader = DataLoader(
        ds,
        batch_size=max(1, preview_n),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    mcfg = cfg["model"]
    torch_dtype = dtype_from_str(mcfg.get("dtype", "float32"))
    model = DualBranchSDVAE(
        pretrained_model=mcfg["pretrained"],
        subfolder=mcfg.get("vae_subfolder", "vae"),
        torch_dtype=torch_dtype,
        local_files_only=bool(mcfg.get("local_files_only", False)),
        ir_out_channels=int(mcfg.get("ir_out_channels", 1)),
    ).to(device)

    lr = float(cfg["train"]["lr"])
    ir_mult = float(cfg["train"].get("lr_ir_mult", 1.0))
    ir_head_params = list(model.decoder_ir.parameters()) + list(model.ir_latent_proj.parameters())
    base_params = (
        list(model.encoder.parameters())
        + list(model.quant_conv.parameters())
        + list(model.post_quant_conv.parameters())
        + list(model.decoder_vis.parameters())
    )
    opt = torch.optim.AdamW(
        [
            {"params": base_params, "lr": lr},
            {"params": ir_head_params, "lr": lr * ir_mult},
        ],
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )

    use_amp = bool(cfg["train"].get("use_amp", True)) and device.type == "cuda"
    amp_dtype = dtype_from_str(cfg["train"].get("amp_dtype", "float16"))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loss_cfg = cfg["loss"]
    # LPIPS for the VIS/IR perceptual losses.
    if float(loss_cfg.get("lambda_lpips_vis", 0.0)) > 0 or float(loss_cfg.get("lambda_lpips_ir", 0.0)) > 0:
        # Perceptual loss requested but lpips not importable -> hard fail
        # rather than degrade silently to L1-only training.
        if lpips_lib is None:
            raise RuntimeError(
                "Config enables LPIPS perceptual loss but `import lpips` failed; refusing to train."
                f"\n  Original error: {LPIPS_IMPORT_ERROR!r}"
                "\n  Fix: pip install lpips."
            )
        lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
        for _p in lpips_fn.parameters():
            _p.requires_grad_(False)
        print("[vae] LPIPS(alex) loaded for VIS/IR perceptual loss")
    else:
        lpips_fn = None

    out_root = Path(cfg["output"]["root"])
    out_root.mkdir(parents=True, exist_ok=True)
    preview_dir = out_root / "recon_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_root / "train_loss_by_epoch.csv"

    epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    recon_preview_every = int(cfg["train"].get("recon_preview_every_epochs", 0))
    recon_preview_every_steps = int(cfg["train"].get("recon_preview_every_steps", 0))

    def dump_recon_preview(filename: str) -> None:
        """Save a VIS-GT | VIS-recon | IR-GT | IR-recon grid to recon_previews/."""
        was_training = model.training
        model.eval()
        with torch.no_grad():
            x_vis_p, x_ir_p, _n, _c = next(iter(preview_loader))
            x_vis_p = x_vis_p.to(device)
            x_ir_p = x_ir_p.to(device)
            out_p = model.forward_fused(x_vis_p, x_ir_p, sample_latent=False)
            vis_hat_p = out_p["vis_hat"]
            ir_hat_p = out_p["ir_hat"]
            x_ir_show = x_ir_p.repeat(1, 3, 1, 1) if x_ir_p.shape[1] == 1 else x_ir_p
            ir_hat_show = ir_hat_p.repeat(1, 3, 1, 1) if ir_hat_p.shape[1] == 1 else ir_hat_p
            rows = []
            n_show = min(preview_n, x_vis_p.shape[0])
            for i in range(n_show):
                rows.append((x_vis_p[i : i + 1].clamp(-1, 1) + 1) * 0.5)
                rows.append((vis_hat_p[i : i + 1].clamp(-1, 1) + 1) * 0.5)
                rows.append((x_ir_show[i : i + 1].clamp(-1, 1) + 1) * 0.5)
                rows.append((ir_hat_show[i : i + 1].clamp(-1, 1) + 1) * 0.5)
            grid = torch.cat(rows, dim=0)
            save_image(grid, preview_dir / filename, nrow=4)
        if was_training:
            model.train()

    global_step = 0
    for epoch in range(epochs):
        model.train()
        ep_sums: Dict[str, float] = {}
        ep_count = 0
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{epochs}")
        for x_vis, x_ir, names, caps in pbar:
            x_vis = x_vis.to(device, non_blocking=True)
            x_ir = x_ir.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model.forward_fused(x_vis, x_ir, sample_latent=True)
                l_kl = model.kl_loss(out["posterior_fused"])

                l_vis = F.l1_loss(out["vis_hat"], x_vis)
                ir_gt = x_ir if x_ir.shape[1] == 1 else rgb_to_gray(x_ir)
                l_ir = F.l1_loss(out["ir_hat"], ir_gt)

                # perceptual (LPIPS, both modalities)
                l_lpips_vis = torch.zeros((), device=device, dtype=l_vis.dtype)
                l_lpips_ir = torch.zeros((), device=device, dtype=l_vis.dtype)
                if lpips_fn is not None:
                    l_lpips_vis = lpips_fn(out["vis_hat"].float(), x_vis.float()).mean()
                    ir_p = out["ir_hat"].repeat(1, 3, 1, 1) if out["ir_hat"].shape[1] == 1 else out["ir_hat"]
                    ir_g_lp = ir_gt.repeat(1, 3, 1, 1) if ir_gt.shape[1] == 1 else ir_gt
                    l_lpips_ir = lpips_fn(ir_p.float(), ir_g_lp.float()).mean()

                # structural (SSIM, both modalities)
                l_ssim = ssim_loss_gray(rgb_to_gray(out["vis_hat"]), rgb_to_gray(x_vis)) + \
                    ssim_loss_gray(out["ir_hat"], ir_gt)

                # thermal-distribution matching (IR only)
                l_dist = thermal_distribution_loss_gray(out["ir_hat"], ir_gt)

                loss = (
                    float(loss_cfg.get("lambda_vis", 1.0)) * l_vis
                    + float(loss_cfg.get("lambda_ir", 1.0)) * l_ir
                    + float(loss_cfg.get("lambda_kl", 1e-6)) * l_kl
                    + float(loss_cfg.get("lambda_lpips_vis", 0.0)) * l_lpips_vis
                    + float(loss_cfg.get("lambda_lpips_ir", 0.0)) * l_lpips_ir
                    + float(loss_cfg.get("lambda_ssim", 0.0)) * l_ssim
                    + float(loss_cfg.get("lambda_thermal_dist", 0.0)) * l_dist
                )
                losses = {
                    "loss": loss,
                    "l_vis": l_vis.detach(),
                    "l_ir": l_ir.detach(),
                    "l_kl": l_kl.detach(),
                    "l_lpips_vis": l_lpips_vis.detach(),
                    "l_lpips_ir": l_lpips_ir.detach(),
                    "l_ssim": l_ssim.detach(),
                    "l_dist": l_dist.detach(),
                }

            ep_count += 1
            with torch.no_grad():
                ep_sums["_total_loss"] = ep_sums.get("_total_loss", 0.0) + float(loss.item())
                for _k, _t in losses.items():
                    if _k == "loss" or not isinstance(_t, torch.Tensor) or _t.numel() != 1:
                        continue
                    ep_sums[str(_k)] = ep_sums.get(str(_k), 0.0) + float(_t.item())

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            global_step += 1
            if global_step % log_every == 0:
                with torch.no_grad():
                    pv = psnr_approx(out["vis_hat"], x_vis)
                    pi = psnr_approx(rgb_to_gray(out["ir_hat"]), rgb_to_gray(x_ir))
                    pbar.set_postfix(
                        loss=float(loss.item()),
                        psnr_v=pv,
                        psnr_i=pi,
                        lv=float(losses["l_vis"].item()),
                        li=float(losses["l_ir"].item()),
                        kl=float(losses["l_kl"].item()),
                    )

            if save_every > 0 and global_step % save_every == 0:
                torch.save(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "config": cfg,
                    },
                    out_root / f"ckpt_step_{global_step}.pt",
                )

            if (
                recon_preview_every_steps > 0
                and global_step % recon_preview_every_steps == 0
            ):
                fn = f"step_{global_step:07d}.png"
                dump_recon_preview(fn)
                print(f"[recon preview] saved {preview_dir / fn}")

        # end of epoch: loss summary + CSV
        if ep_count > 0:
            avg = {k: ep_sums[k] / ep_count for k in ep_sums}
            print(f"\n========== epoch {epoch + 1}/{epochs} mean losses (n={ep_count} steps) ==========")
            print(f"  total_loss: {avg['_total_loss']:.6f}")
            print(
                f"  L_vis: {avg.get('l_vis', 0):.6f}  L_ir: {avg.get('l_ir', 0):.6f}  KL: {avg.get('l_kl', 0):.6f}  "
                f"LPIPS_v: {avg.get('l_lpips_vis', 0):.6f}  LPIPS_i: {avg.get('l_lpips_ir', 0):.6f}  "
                f"SSIM: {avg.get('l_ssim', 0):.6f}  thermal: {avg.get('l_dist', 0):.6f}"
            )
            row = {"epoch": epoch + 1}
            row.update({f"avg_{k}": float(f"{v:.8g}") for k, v in avg.items()})
            file_exists = log_csv.is_file()
            with open(log_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
                if not file_exists:
                    w.writeheader()
                w.writerow(row)

        torch.save(
            {
                "global_step": global_step,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "config": cfg,
            },
            out_root / "ckpt_last.pt",
        )
        if recon_preview_every > 0 and (epoch + 1) % recon_preview_every == 0:
            fn = f"epoch_{epoch+1:04d}.png"
            dump_recon_preview(fn)
            print(f"[recon preview] saved {preview_dir / fn}")

    print(f"Done. Checkpoints under {out_root}")


if __name__ == "__main__":
    main()
