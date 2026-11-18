"""
DDIM sampling in SD latent space. Shares its logic with the training-time
preview routine so that inference and training previews behave identically
(same scheduler / precision / CFG handling).
"""
from __future__ import annotations

from typing import Sequence

import torch
from diffusers import DDIMScheduler


@torch.inference_mode()
def ddim_sample_scaled_latents(
    *,
    unet,
    tokenizer,
    text_encoder,
    sd_id: str,
    local_files_only: bool,
    device: torch.device,
    prompts: Sequence[str],
    num_inference_steps: int,
    guidance_scale: float,
    img_size: int,
    seed: int,
    seeds: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    Returns latent tensor in **scaled** space (the diffusers SD pipeline
    convention, i.e. before dividing by the VAE scaling_factor for decode).
    If `seeds` is given (one per prompt), each sample gets its own generator;
    otherwise a single generator seeded with `seed` produces the whole batch.
    """
    n = len(prompts)
    if n < 1:
        raise ValueError("prompts must be non-empty")
    if seeds is not None and len(seeds) != n:
        raise ValueError("seeds length must match len(prompts)")

    sched = DDIMScheduler.from_pretrained(
        sd_id, subfolder="scheduler", local_files_only=local_files_only
    )
    unet_dtype = next(unet.parameters()).dtype
    latent_h = img_size // 8
    latent_w = img_size // 8

    tok = tokenizer(
        list(prompts),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    enc = text_encoder(tok.input_ids.to(device))[0].to(dtype=unet_dtype)

    uncond_enc = None
    if guidance_scale > 1.0:
        tok_u = tokenizer(
            [""],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_enc = text_encoder(tok_u.input_ids.to(device))[0].to(dtype=unet_dtype)
        if uncond_enc.shape[0] == 1:
            uncond_enc = uncond_enc.expand(n, -1, -1)

    if seeds is None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        latents = torch.randn(
            (n, 4, latent_h, latent_w),
            generator=gen,
            device=device,
            dtype=unet_dtype,
        )
    else:
        rows = []
        for i in range(n):
            gen_i = torch.Generator(device=device).manual_seed(int(seeds[i]))
            rows.append(
                torch.randn(
                    (1, 4, latent_h, latent_w),
                    generator=gen_i,
                    device=device,
                    dtype=unet_dtype,
                )
            )
        latents = torch.cat(rows, dim=0)

    sched.set_timesteps(int(num_inference_steps), device=device)
    sig = getattr(sched, "init_noise_sigma", None)
    if sig is not None:
        latents = latents * float(sig)

    for t in sched.timesteps:
        t_int = int(t) if not torch.is_tensor(t) else int(t.item())
        ts = torch.full((n,), t_int, device=device, dtype=torch.long)
        noise_pred = unet(
            latents,
            ts,
            encoder_hidden_states=enc,
        ).sample
        if guidance_scale > 1.0 and uncond_enc is not None:
            nu = unet(
                latents,
                ts,
                encoder_hidden_states=uncond_enc,
            ).sample
            noise_pred = nu + float(guidance_scale) * (noise_pred - nu)
        step_out = sched.step(noise_pred, t_int, latents)
        latents = step_out.prev_sample

    return latents
