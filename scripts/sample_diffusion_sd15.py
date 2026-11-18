#!/usr/bin/env python3
"""
Batch sampling after Stage 2: DDIM latent sampling + fine-tuned UNet weights,
then dual-decoder VAE decoding into side-by-side VIS | IR pair images.
FP32 UNet by default; `--fp16` saves memory but may deviate slightly from FP32.

  # Generate 16 pairs (batch 4)
  python scripts/sample_diffusion_sd15.py \
    --sd-id runwayml/stable-diffusion-v1-5 \
    --dual-ckpt checkpoints/m3fd/dual_vae_v2.pt \
    --unet-ckpt checkpoints/m3fd/unet_v2.pt \
    --prompt-file prompts/m3fd_clean_prompts_3000.txt --prompt-mode cycle \
    --num-samples 16 --batch-size 4 \
    --steps 30 --guidance-scale 7.5 --seed 42 \
    --out-dir runs/sample_vis_ir

  # Single image to a given path
  python scripts/sample_diffusion_sd15.py \
    --dual-ckpt ... --unet-ckpt ... \
    --num-samples 1 --out single.png
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sd_dual.dual_vae import DualBranchSDVAE
from sd_dual.sd15_ddim_sample import ddim_sample_scaled_latents

try:
    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install -r requirements.txt") from e


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-generate VIS|IR pair images (left: visible, right: infrared)")
    ap.add_argument(
        "--sd-id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion 1.5 model id (HF hub id or local directory)",
    )
    ap.add_argument("--offline", action="store_true", help="Force local files only (no hub download)")
    ap.add_argument("--dual-ckpt", type=str, required=True)
    ap.add_argument(
        "--ir-out-channels",
        type=int,
        default=1,
        choices=[1, 3],
        help="Must match the dual VAE checkpoint",
    )
    ap.add_argument("--unet-ckpt", type=str, default="", help="Fine-tuned UNet .pt (dict with an 'unet' key)")
    ap.add_argument(
        "--prompt",
        type=str,
        default="Outdoor scene, sharp clear subjects, no geometric distortion.",
    )
    ap.add_argument(
        "--prompt-file",
        type=str,
        default="",
        help="Optional: text file with one prompt per line; overrides the single --prompt",
    )
    ap.add_argument(
        "--prompt-mode",
        type=str,
        default="cycle",
        choices=["cycle", "random"],
        help="How prompts are drawn from --prompt-file: cycle in order, or random",
    )
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Output resolution (match training data.img_size; must be divisible by 8)",
    )
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="FP16 UNet/text-encoder sampling (saves memory; may differ slightly from FP32)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Image i uses seed+i (reproducible)",
    )
    ap.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=1,
        help="Total number of pairs (each output is one VIS|IR side-by-side image)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Pairs per pipeline call; lower to 1-2 if GPU memory is tight",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory when num-samples>1; default runs/sample_vis_ir",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output file path, valid only when num-samples=1 (default runs/sample_vis_ir/pair_0000.png)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Shard index (0-based): this shard handles global indices shard, shard+n_shards, ...",
    )
    ap.add_argument(
        "--n-shards",
        type=int,
        default=1,
        help="Total number of shards; sharding keeps prompt indices / seeds / filenames identical to a single process",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    lfo = bool(args.offline)
    n = max(1, int(args.num_samples))
    bs_cap = max(1, int(args.batch_size))

    prompts = None
    if args.prompt_file:
        pf = Path(args.prompt_file)
        if not pf.is_file():
            raise FileNotFoundError(f"prompt file not found: {pf}")
        lines = [ln.strip() for ln in pf.read_text(encoding="utf-8").splitlines()]
        prompts = [ln for ln in lines if ln]
        if not prompts:
            raise ValueError("prompt file has no non-empty lines.")
        print(f"Loaded {len(prompts)} prompts from {pf}")

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / "sample_vis_ir"
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.dual_ckpt, map_location=device, weights_only=False)
    dual = DualBranchSDVAE(
        pretrained_model=args.sd_id,
        subfolder="vae",
        local_files_only=lfo,
        ir_out_channels=args.ir_out_channels,
    ).to(device)
    info = dual.load_state_dict(ck["state_dict"], strict=False)
    if info.unexpected_keys:
        print(f"[warn] unexpected keys in dual ckpt: {info.unexpected_keys}")
    if info.missing_keys:
        print(f"[warn] missing keys when loading dual ckpt: {info.missing_keys}")
    dual.eval()
    dec_dtype = dual.decoder_vis.conv_in.weight.dtype

    tokenizer = CLIPTokenizer.from_pretrained(
        args.sd_id, subfolder="tokenizer", local_files_only=lfo
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd_id, subfolder="text_encoder", local_files_only=lfo
    ).to(device)
    text_encoder.eval()

    unet = UNet2DConditionModel.from_pretrained(
        args.sd_id, subfolder="unet", local_files_only=lfo
    ).to(device)
    unet.eval()

    if args.fp16 and device.type == "cuda":
        text_encoder = text_encoder.to(dtype=torch.float16)
        unet = unet.to(dtype=torch.float16)

    if args.unet_ckpt:
        unet_ckpt = torch.load(args.unet_ckpt, map_location=device, weights_only=False)
        if "unet" in unet_ckpt:
            unet.load_state_dict(unet_ckpt["unet"], strict=True)
        else:
            unet.load_state_dict(unet_ckpt, strict=True)

    from torchvision.utils import save_image

    # Sharding: shard k handles global indices k, k+n_shards, ... — prompt
    # indices / seeds / filenames stay identical to a single-process run.
    idxs = list(range(args.shard, n, max(1, args.n_shards)))
    print(f"Shard {args.shard}/{args.n_shards}: {len(idxs)} images (of {n})")
    saved = 0
    while saved < len(idxs):
        bs = min(bs_cap, len(idxs) - saved)
        chunk = idxs[saved : saved + bs]
        seeds = [args.seed + g for g in chunk]
        if prompts is None:
            batch_prompts = [args.prompt] * bs
        else:
            if args.prompt_mode == "cycle":
                batch_prompts = [prompts[g % len(prompts)] for g in chunk]
            else:
                batch_prompts = [random.choice(prompts) for _ in range(bs)]
        latents = ddim_sample_scaled_latents(
            unet=unet,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            sd_id=args.sd_id,
            local_files_only=lfo,
            device=device,
            prompts=batch_prompts,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            img_size=args.img_size,
            seed=args.seed,
            seeds=seeds,
        )
        latents = latents.to(dtype=dec_dtype)
        z = latents / dual.scaling_factor
        with torch.no_grad():
            vis = dual.decode_vis(z)
            ir = dual.decode_ir(z)
            if ir.shape[1] == 1:
                ir = ir.repeat(1, 3, 1, 1)

        vis_n = (vis.clamp(-1, 1) + 1) * 0.5
        ir_n = (ir.clamp(-1, 1) + 1) * 0.5

        for i, idx in enumerate(chunk):
            pair = torch.cat([vis_n[i : i + 1], ir_n[i : i + 1]], dim=3)
            if n == 1 and max(1, args.n_shards) == 1:
                out_path = Path(args.out) if args.out else out_dir / "pair_0000.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(pair, str(out_path))
                print(f"Saved {out_path} (left=VIS, right=IR)")
            else:
                out_path = out_dir / f"pair_{idx:04d}.png"
                save_image(pair, str(out_path))
                print(f"Saved {out_path}")

        saved += bs

    if n > 1:
        print(f"Done. {len(idxs)} images (shard {args.shard}) under {out_dir}")


if __name__ == "__main__":
    main()
