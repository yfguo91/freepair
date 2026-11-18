"""
Shared SD1.5 encoder + dual decoders (VIS / IR), compatible with the Stable
Diffusion latent space.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn as nn

from diffusers import AutoencoderKL


def fused_vis_ir_input(x_vis: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
    """
    Pixel-domain fusion operator F (Eq. 2): replicate the single-channel
    infrared image to three channels and average it with the visible image.

    x_vis: [B,3,H,W], x_ir: [B,1,H,W] or [B,3,H,W], values in [-1,1].
    Returns [B,3,H,W], the input to the shared encoder.
    """
    if x_ir.shape[1] == 1:
        x_ir_3 = x_ir.repeat(1, 3, 1, 1)
    elif x_ir.shape[1] == 3:
        x_ir_3 = x_ir
    else:
        raise ValueError(f"x_ir must have 1 or 3 channels, got shape {x_ir.shape}")
    return (x_vis + x_ir_3) * 0.5


class DualBranchSDVAE(nn.Module):
    """
    One SD VAE encoder + quant_conv, a shared post_quant_conv, and two
    independent decoders (VIS / IR). A learnable 1x1 convolution P (the IR
    latent projection, identity-initialized) re-mixes the latent before the IR
    decoder so the IR branch can depart slightly from the VIS latent basis
    without disturbing the latent space seen by the diffusion UNet.

    Latents are in the *unscaled* VAE space (before scaling_factor).
    """

    def __init__(
        self,
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        subfolder: str = "vae",
        torch_dtype: torch.dtype | None = None,
        local_files_only: bool = False,
        ir_out_channels: int = 1,
    ) -> None:
        super().__init__()
        kwargs: Dict[str, Any] = {"subfolder": subfolder}
        if local_files_only:
            kwargs["local_files_only"] = True
        ae = AutoencoderKL.from_pretrained(pretrained_model, **kwargs)
        if torch_dtype is not None:
            ae = ae.to(dtype=torch_dtype)

        self.encoder = ae.encoder
        self.quant_conv = ae.quant_conv
        self.post_quant_conv = ae.post_quant_conv
        self.decoder_vis = copy.deepcopy(ae.decoder)
        self.decoder_ir = copy.deepcopy(ae.decoder)
        self.ir_out_channels = int(ir_out_channels)

        # IR latent projection P: per-channel 1x1 conv, initialized as identity.
        latent_ch = int(getattr(ae.config, "latent_channels", 4))
        self.ir_latent_proj = nn.Conv2d(latent_ch, latent_ch, kernel_size=1, bias=True)
        with torch.no_grad():
            self.ir_latent_proj.weight.zero_()
            for i in range(latent_ch):
                self.ir_latent_proj.weight[i, i, 0, 0] = 1.0
            self.ir_latent_proj.bias.zero_()

        if self.ir_out_channels not in (1, 3):
            raise ValueError("ir_out_channels must be 1 or 3")
        if self.ir_out_channels == 1:
            self._set_ir_decoder_single_channel()

        # SD convention: the UNet sees latents * scaling_factor
        self.register_buffer(
            "scaling_factor",
            torch.tensor(float(getattr(ae.config, "scaling_factor", 0.18215))),
        )
        self.config = ae.config

    def _set_ir_decoder_single_channel(self) -> None:
        """Give the IR decoder a 1-channel output, initialized from the RGB mean."""
        conv = getattr(self.decoder_ir, "conv_out", None)
        if conv is None or not isinstance(conv, nn.Conv2d):
            raise RuntimeError("decoder_ir has no Conv2d conv_out; cannot set ir_out_channels=1")
        if conv.out_channels == 1:
            return
        new_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=1,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None),
        ).to(device=conv.weight.device, dtype=conv.weight.dtype)
        with torch.no_grad():
            # initialize from the RGB average to keep the SD decoder prior
            new_conv.weight.copy_(conv.weight.mean(dim=0, keepdim=True))
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias.mean(dim=0, keepdim=False).view(1))
        self.decoder_ir.conv_out = new_conv

    def encode_dist(self, x: torch.Tensor):
        """Gaussian posterior q(z|x) for the fused input (diffusers distribution)."""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        return DiagonalGaussianDistribution(moments)

    def latent_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior mean for encode(x), in unscaled VAE space."""
        return self.encode_dist(x).mean

    def latent_mean_fused(self, x_vis: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        """Posterior mean of the fused VIS+IR input, in unscaled VAE space."""
        return self.latent_mean(fused_vis_ir_input(x_vis, x_ir))

    def forward_fused(
        self,
        x_vis: torch.Tensor,
        x_ir: torch.Tensor,
        sample_latent: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode the fused VIS+IR input once; decode both modalities.
        sample_latent: True -> z ~ q(z|x_fused); False -> z = mean(q).
        """
        x_fused = fused_vis_ir_input(x_vis, x_ir)
        post = self.encode_dist(x_fused)
        z = post.sample() if sample_latent else post.mean
        return {
            "vis_hat": self.decode_vis(z),
            "ir_hat": self.decode_ir(z),
            "z": z,
            "posterior_fused": post,
        }

    def decode_vis(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder_vis(z)

    def decode_ir(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        z = self.ir_latent_proj(z)
        return self.decoder_ir(z)

    def kl_loss(self, posterior) -> torch.Tensor:
        """KL(q || N(0,I)), mean over batch."""
        return posterior.kl().mean()
