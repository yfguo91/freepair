"""
Auxiliary infrared prior for epsilon-prediction diffusion training (frozen VAE).

Gradients flow: UNet -> eps_hat -> z_hat_0 (x0-prediction, scaled space)
-> decode through the frozen IR branch -> statistical-moment prior L_P.
"""
from __future__ import annotations

from typing import Any, Dict

import torch


def predict_x0_from_epsilon(
    noisy_latents: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    DDPM x0-prediction: x0 = (x_t - sqrt(1-abar_t) * eps) / sqrt(abar_t),
    in the same (scaled) latent space used by diffusers `add_noise`.
    """
    ab = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sqrt_ab = torch.sqrt(ab.clamp(min=1e-20))
    sqrt_omab = torch.sqrt((1.0 - ab).clamp(min=0.0))
    return (noisy_latents - sqrt_omab * noise_pred) / sqrt_ab


def _to_gray01(ir: torch.Tensor) -> torch.Tensor:
    """ir [B,1or3,H,W] in [-1,1] -> [B,1,H,W] in [0,1]."""
    if ir.shape[1] == 3:
        r, g, b = ir[:, 0:1], ir[:, 1:2], ir[:, 2:3]
        ir = 0.299 * r + 0.587 * g + 0.114 * b
    return (ir + 1.0) * 0.5


def ir_stat_moment_loss_per_sample(ir_pred: torch.Tensor, ir_gt: torch.Tensor) -> torch.Tensor:
    """
    Infrared statistical-moment prior L_P (Eq. 8): per-sample L1 matching of
    (mean, std) between the predicted and ground-truth IR intensity
    distributions, on gray values normalized to [0,1]. Returns shape [B].
    """
    gp = _to_gray01(ir_pred).flatten(1)  # [B, H*W]
    gg = _to_gray01(ir_gt).flatten(1)
    mp, mg = gp.mean(1), gg.mean(1)
    sp, sg = gp.std(1), gg.std(1)
    return (mp - mg).abs() + (sp - sg).abs()


def auxiliary_decode_losses(
    *,
    dual: Any,
    noisy_latents: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    x_ir_gt: torch.Tensor,
    z_clip: float = 10.0,
    aux_timestep_max: int | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Decode the x0-predicted latent through the frozen IR branch and compute the
    statistical-moment prior. If aux_timestep_max is set (threshold tau), the
    prior is only active for samples with t <= tau; the returned value is the
    masked mean (0 when no sample passes the threshold).
    """
    x0_s = predict_x0_from_epsilon(noisy_latents, noise_pred, timesteps, alphas_cumprod)
    z = x0_s / dual.scaling_factor
    if z_clip > 0:
        z = z.clamp(-z_clip, z_clip)
    ir_p = dual.decode_ir(z)

    bsz = timesteps.shape[0]
    device = noisy_latents.device
    dtype = noisy_latents.dtype
    if aux_timestep_max is None:
        mask = torch.ones(bsz, device=device, dtype=dtype)
    else:
        mask = (timesteps <= int(aux_timestep_max)).to(dtype=dtype)

    per = ir_stat_moment_loss_per_sample(ir_p, x_ir_gt)
    if mask.sum() < 0.5:
        stat_val = torch.zeros((), device=device, dtype=dtype)
    else:
        stat_val = (per * mask).sum() / mask.sum().clamp(min=1.0)
    return {"ir_stat": stat_val}


def fused_latents_for_training(
    dual: Any, x_vis: torch.Tensor, x_ir: torch.Tensor
) -> torch.Tensor:
    """Scaled clean latent z0 = E(F(x_vis, x_ir)) * scaling_factor (Eq. 6)."""
    return dual.latent_mean_fused(x_vis, x_ir) * dual.scaling_factor
