"""Temporal window sampling utilities used by the GigaWorld-0 data pipeline.

The training transform receives videos at their native frame rate, while the
world model is conditioned on a fixed output ``fps``.  Sampling a fixed number
of frames with ``linspace`` makes the temporal stride depend on the duration of
each episode.  The helpers in this module instead sample timestamps at a
constant rate and make any out-of-range tail explicit through a validity mask.

The module intentionally only depends on NumPy so that the sampling policy can
be tested without importing the heavyweight model and video-reader packages.
"""

from __future__ import annotations
import math
from typing import Any

import numpy as np


def _validate_video_sampling_args(video_length: int, num_frames: int, target_fps: float) -> None:
    """Validate arguments shared by the temporal sampling functions."""

    if isinstance(video_length, bool) or int(video_length) != video_length or video_length <= 0:
        raise ValueError(f"video_length must be a positive integer, got {video_length!r}")
    if isinstance(num_frames, bool) or int(num_frames) != num_frames or num_frames <= 0:
        raise ValueError(f"num_frames must be a positive integer, got {num_frames!r}")
    if not math.isfinite(float(target_fps)) or float(target_fps) <= 0:
        raise ValueError(f"target_fps must be a finite positive number, got {target_fps!r}")


def _draw_start(rng: Any, upper_bound: int) -> int:
    """Draw an integer in ``[0, upper_bound)`` from common RNG interfaces.

    ``random.Random``, NumPy's ``Generator``/``RandomState`` and the NumPy
    module itself are all used by data-loader workers in practice.  Keeping the
    small adapter here lets callers inject a seeded RNG without changing the
    transform's existing randomness behaviour.
    """

    if upper_bound <= 1:
        return 0
    if rng is None:
        rng = np.random
    if hasattr(rng, "integers"):
        return int(rng.integers(0, upper_bound))
    if hasattr(rng, "randrange"):
        return int(rng.randrange(upper_bound))
    if hasattr(rng, "randint"):
        # NumPy uses an exclusive high bound; Python's ``randint`` uses an
        # inclusive one.  Prefer the two-argument form and handle both APIs.
        try:
            return int(rng.randint(0, upper_bound))
        except TypeError:
            return int(rng.randint(upper_bound - 1))
    raise TypeError("rng must provide integers(), randrange(), or randint()")


def sample_temporal_window(
    video_length: int,
    num_frames: int,
    target_fps: float,
    source_fps: float | None = None,
    *,
    start_frame: int | None = None,
    random_start: bool = True,
    rng: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-rate frame indices and a real-frame validity mask.

    Frame ``i`` is selected at timestamp ``i / target_fps`` from the window;
    timestamps are converted to the nearest source frame using cumulative
    rounding.  A random start is selected only from windows that fit entirely
    in the episode.  An explicitly supplied ``start_frame`` may extend past
    the episode end, which is useful for deterministic autoregressive windows.
    Such positions are edge-padded by repeating the last frame and marked
    ``False`` in the returned mask.

    Args:
        video_length: Number of frames in the source video.
        num_frames: Number of frames required by the model.
        target_fps: Temporal rate expected by the model.
        source_fps: Native source-video rate.  Defaults to ``target_fps`` when
            metadata is unavailable, preserving one source frame per output
            frame.
        start_frame: Optional source-frame index for the beginning of the
            window.  If omitted, a random valid start (or zero when
            ``random_start=False``) is used.
        random_start: Whether to sample a random valid start when
            ``start_frame`` is omitted.
        rng: Optional random-number generator implementing a common Python or
            NumPy RNG interface.

    Returns:
        ``(indices, valid_mask)`` where ``indices`` is an ``int64`` array of
        length ``num_frames`` (edge-clipped to the source range) and
        ``valid_mask`` is a boolean array identifying non-padded positions.
    """

    _validate_video_sampling_args(video_length, num_frames, target_fps)
    video_length = int(video_length)
    num_frames = int(num_frames)

    if source_fps is None:
        source_fps = float(target_fps)
    if not math.isfinite(float(source_fps)) or float(source_fps) <= 0:
        raise ValueError(f"source_fps must be a finite positive number, got {source_fps!r}")
    source_fps = float(source_fps)

    # Convert timestamps to source-frame offsets with cumulative rounding.  It
    # preserves the requested duration and avoids the episode-length-dependent
    # stride introduced by np.linspace(0, video_length - 1, num_frames).
    source_step = source_fps / float(target_fps)
    offsets = np.rint(np.arange(num_frames, dtype=np.float64) * source_step).astype(np.int64)
    # Rounding should be monotonic for positive timestamps, but enforcing it
    # protects against unusual floating-point ratios and documents the index
    # contract expected by video readers.
    offsets = np.maximum.accumulate(offsets)

    window_span = int(offsets[-1])
    max_start = max(video_length - 1 - window_span, 0)
    if start_frame is None:
        start = _draw_start(rng, max_start + 1) if random_start else 0
    else:
        if isinstance(start_frame, bool) or int(start_frame) != start_frame:
            raise ValueError(f"start_frame must be an integer, got {start_frame!r}")
        start = int(start_frame)
        if start < 0 or start >= video_length:
            raise ValueError(f"start_frame must be in [0, {video_length}), got {start}")

    raw_indices = start + offsets
    valid_mask = raw_indices < video_length
    indices = np.minimum(raw_indices, video_length - 1).astype(np.int64, copy=False)
    return indices, valid_mask.astype(bool, copy=False)


def sample_uniform_window(video_length: int, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the original full-episode uniform sampling policy.

    This compatibility path is intentionally kept separate from
    :func:`sample_temporal_window`.  Uniform sampling stretches short videos
    over the whole model window and therefore has no padded positions; its
    validity mask is all ``True`` because every returned index references a
    real frame (possibly more than once).
    """

    _validate_video_sampling_args(video_length, num_frames, target_fps=1.0)
    indices = np.linspace(0, int(video_length) - 1, int(num_frames), dtype=np.int64)
    return indices, np.ones(int(num_frames), dtype=bool)


def frame_mask_to_latent_mask(frame_mask: np.ndarray, temporal_factor: int) -> np.ndarray:
    """Convert frame validity to conservative VAE-latent validity.

    GigaWorld-0's temporal VAE represents the first frame separately and then
    consumes groups of ``temporal_factor`` frames.  A latent is marked valid
    only when every source frame in its group is real; this prevents edge-padded
    frames from contributing to the diffusion loss while retaining the first
    conditioning frame.
    """

    mask = np.asarray(frame_mask, dtype=bool)
    if mask.ndim != 1 or mask.size == 0:
        raise ValueError(f"frame_mask must be a non-empty 1D array, got shape {mask.shape}")
    if isinstance(temporal_factor, bool) or int(temporal_factor) != temporal_factor or temporal_factor <= 0:
        raise ValueError(f"temporal_factor must be a positive integer, got {temporal_factor!r}")
    temporal_factor = int(temporal_factor)
    if (mask.size - 1) % temporal_factor != 0:
        raise ValueError("frame_mask length must be 1 + k * temporal_factor; " f"got length={mask.size}, temporal_factor={temporal_factor}")

    latent_mask = np.empty(1 + (mask.size - 1) // temporal_factor, dtype=bool)
    latent_mask[0] = mask[0]
    if latent_mask.size > 1:
        latent_mask[1:] = mask[1:].reshape(-1, temporal_factor).all(axis=1)
    return latent_mask
