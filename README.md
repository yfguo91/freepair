# FreePair: Reference-free Joint Generation of Aligned Visible–Infrared Image Pairs

Official code for training and sampling **FreePair**, a Stable-Diffusion-1.5-based
model that generates **spatially aligned visible (VIS) / infrared (IR) image pairs
without any reference image**: one shared latent is sampled by the diffusion UNet
and decoded by two modality-specific VAE decoders, so pixel-level alignment comes
for free.

Trained checkpoints are hosted on HuggingFace (see [Checkpoints](#checkpoints));
this repository contains the training, data-preparation and inference code.

## Method overview

- **Stage 1 — shared-latent dual-branch VAE** (`sd_dual/dual_vae.py`,
  `scripts/train_dual_vae_sd15.py`): a single SD VAE encoder embeds the fused
  input `F(x_vis, x_ir)` (Eq. 2: IR replicated to 3 channels and averaged with
  the VIS image); two decoders reconstruct both modalities from the same latent,
  with a learnable identity-initialized 1×1 projection `P` in front of the IR
  decoder (Eq. 5). The loss (Eq. 4) combines L1, KL, LPIPS, SSIM and a
  thermal-distribution matching term (first two intensity moments, Eq. 5).
- **Stage 2 — text-conditioned diffusion** (`scripts/train_unet_sd15.py`): the
  SD1.5 UNet is fine-tuned (full parameters, ε-prediction) on the fused latent
  `z0 = E(F(x_vis, x_ir)) · s` (Eq. 6) with CLIP caption conditioning. Besides
  ε-MSE, the **infrared statistical-moment prior L_P** (Eq. 8) decodes the
  x0-prediction through the frozen IR branch and matches the per-sample
  (mean, std) of the predicted IR intensity to the ground truth.
- **Inference** (`scripts/sample_diffusion_sd15.py`): DDIM (30 steps) +
  classifier-free guidance (7.5); one latent is decoded twice into a
  side-by-side VIS | IR pair.

## Installation

```bash
pip install -r requirements.txt
```

You also need the Stable Diffusion 1.5 base weights (VAE / UNet / text encoder /
tokenizer / scheduler), either from the HF hub (`runwayml/stable-diffusion-v1-5`)
or from a local directory passed as `--sd-id <path>` (also set
`paths.sd_id` / `local_files_only` in the configs).

## Data preparation

Paired VIS/IR datasets with matching filenames, resized to 512×512 at load time:

1. **M3FD**: place the `vi/` and `ir/` folders under `data/M3FD/`.

2. **LLVIP**: flatten the official train/test splits first:
   
   ```bash
   python scripts/prepare_llvip_pairs.py --llvip-root data/LLVIP --out-root data/LLVIP_pair
   ```

3. **Captions** (Stage 2 conditions on CLIP text embeddings). Caption the VIS
   side with any captioner (we used BLIP) and save a JSON mapping
   `{"<image filename>": "<caption>", ...}`:
   
   ```bash
   python scripts/caption_vis_dataset.py --vis-dir data/M3FD/vi --out data/M3FD/vi_captions.json
   ```
   
   `scripts/make_prompts_from_captions.py` builds the sampling prompt list from
   the same JSON.

## Training

```bash
# Stage 1: dual-branch VAE (~10 epochs)
python scripts/train_dual_vae_sd15.py --config configs/train_vae_m3fd_v2.yaml

# Stage 2: UNet fine-tuning (15k steps, full parameters + L_P prior)
python scripts/train_unet_sd15.py --config configs/train_unet_m3fd_v2.yaml
```

Swap `m3fd` for `llvip` (and adjust the `data:` paths) for LLVIP. Stage 2
expects the Stage-1 checkpoint at `checkpoints/<dataset>/dual_vae_v2.pt`
(download from HuggingFace, see below, or point `paths.dual_vae_ckpt` at your
own Stage-1 output).

## Inference

```bash
python scripts/sample_diffusion_sd15.py \
  --sd-id runwayml/stable-diffusion-v1-5 \
  --dual-ckpt checkpoints/m3fd/dual_vae_v2.pt \
  --unet-ckpt checkpoints/m3fd/unet_v2.pt \
  --prompt-file prompts/m3fd_clean_prompts_3000.txt --prompt-mode cycle \
  --num-samples 16 --batch-size 4 \
  --steps 30 --guidance-scale 7.5 --seed 42 \
  --out-dir runs/sample_vis_ir
```

Each output `pair_XXXX.png` is a side-by-side image (left: VIS, right: IR).
Reproduction protocol for the numbers below: seed 42 (image *i* uses seed *i*),
DDIM 30 steps, CFG 7.5, 512×512. The sampler also supports sharding
(`--shard k --n-shards m`) with identical prompts/seeds/filenames to a
single-process run.

## Checkpoints

Hosted on HuggingFace (the UNet is a full-parameter fine-tune, ~3.4 GB, which
exceeds GitHub's file-size limit):

| Dataset | Files                                            |
| ------- | ------------------------------------------------ |
| M3FD    | `dual_vae_v2.pt` (532 MB), `unet_v2.pt` (3.4 GB) |
| LLVIP   | `dual_vae_v2.pt` (532 MB), `unet_v2.pt` (3.4 GB) |

SHA-256 checksums for all four files are in [checkpoints/CHECKSUMS.txt](checkpoints/CHECKSUMS.txt).

## Repository layout

```
sd_dual/
  dual_vae.py            shared-encoder dual-decoder VAE (+ fusion operator F)
  data_pair.py           paired VIS/IR dataset
  diffusion_aux_loss.py  L_P prior + fused-latent construction
  sd15_ddim_sample.py    DDIM sampling in SD latent space (shared by training previews)
scripts/
  caption_vis_dataset.py      caption the VIS side (BLIP)
  make_prompts_from_captions.py  build sampling prompt lists from caption JSON
  prepare_llvip_pairs.py      flatten LLVIP train/test into paired dirs
  train_dual_vae_sd15.py      Stage 1 training
  train_unet_sd15.py          Stage 2 training (UNet + L_P prior)
  sample_diffusion_sd15.py    pair sampling (shard-safe)
configs/                  Stage-1/Stage-2 configs for M3FD and LLVIP
prompts/                  caption-derived sampling prompt lists (3,000 lines each)
```

## Notes

- The released UNet checkpoints were trained with `aux_timestep_max: null`
  (no timestep threshold on L_P), which is the setting reflected in the
  configs.
- `--fp16` sampling saves memory but may deviate slightly from the FP32
  numbers above.
