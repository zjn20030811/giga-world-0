"""Loss helpers for variable-length temporal training windows."""

from __future__ import annotations

import torch


def compute_masked_edm_loss(edm_loss, denoised_latents: torch.Tensor, latent_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the EDM objective while ignoring padded temporal latents.

    ``EDMLoss.compute_loss`` averages over every latent element. A fixed-rate
    window can extend beyond a short episode, so edge-padded latents must not
    contribute to that average. The all-ones path below is mathematically
    identical to ``EDMLoss.compute_loss`` and keeps compatibility with custom
    transforms that do not emit a mask.
    """

    if latent_mask is None:
        return edm_loss.compute_loss(denoised_latents)

    target = edm_loss.latents
    prediction = denoised_latents.reshape(target.shape)
    loss_weight = edm_loss.get_loss_weight()
    while loss_weight.ndim < prediction.ndim:
        loss_weight = loss_weight.unsqueeze(-1)
    weighted_error = loss_weight * (prediction - target) ** 2

    if latent_mask.ndim == 2:
        # Collated transform output: (batch, latent_time).
        mask = latent_mask[:, None, :, None, None]
    elif latent_mask.ndim == 5:
        # Also accept a pre-shaped mask from custom data pipelines.
        mask = latent_mask
        if mask.shape[1] != 1 or mask.shape[3:] != (1, 1):
            raise ValueError('5D latent_mask must have singleton channel/spatial dimensions ' f'(B, 1, T, 1, 1), got {tuple(mask.shape)}')
    else:
        raise ValueError(f'latent_mask must have shape (B, T) or (B, 1, T, 1, 1), got {latent_mask.shape}')
    if mask.shape[0] != weighted_error.shape[0] or mask.shape[2] != weighted_error.shape[2]:
        raise ValueError(
            'latent_mask temporal shape does not match latent tensor: ' f'mask={tuple(mask.shape)}, latents={tuple(weighted_error.shape)}'
        )
    mask = mask.to(device=weighted_error.device, dtype=weighted_error.dtype)

    # Normalize by the number of valid elements per sample, preserving the
    # scale of the original mean reduction even when samples have different
    # amounts of right-padding.
    valid_elements = mask.sum(dim=(1, 2, 3, 4))
    elements_per_latent = weighted_error.shape[1] * weighted_error.shape[3] * weighted_error.shape[4]
    denominator = (valid_elements * elements_per_latent).clamp_min(1.0)
    return (weighted_error * mask).sum(dim=(1, 2, 3, 4)) / denominator
