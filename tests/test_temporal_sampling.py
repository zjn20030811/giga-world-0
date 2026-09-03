"""Regression tests for fixed-rate temporal windows and masked EDM loss."""

from __future__ import annotations
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    """Load a small module without importing optional model dependencies."""

    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


temporal_sampling = _load_module('giga_world_0_temporal_sampling_test', 'giga_world_0/temporal_sampling.py')
loss_utils = _load_module('giga_world_0_loss_utils_test', 'giga_world_0/loss_utils.py')


class _FakeEDMLoss:
    """Minimal stateful EDM object for testing the reduction helper."""

    def __init__(self, latents: torch.Tensor, weight: torch.Tensor):
        self.latents = latents
        self.weight = weight

    def get_loss_weight(self) -> torch.Tensor:
        return self.weight

    def compute_loss(self, denoised_latents: torch.Tensor) -> torch.Tensor:
        error = (denoised_latents.reshape(self.latents.shape) - self.latents) ** 2
        weight = self.weight
        while weight.ndim < error.ndim:
            weight = weight.unsqueeze(-1)
        return (weight * error).reshape(error.shape[0], -1).mean(dim=1)


def test_fixed_fps_uses_source_to_target_stride():
    indices, valid = temporal_sampling.sample_temporal_window(
        video_length=300,
        num_frames=5,
        target_fps=16,
        source_fps=30,
        start_frame=10,
    )

    # Cumulative rounding of [0, 1.875, 3.75, 5.625, 7.5] gives this stable
    # source-frame schedule, independent of the full episode length.
    np.testing.assert_array_equal(indices, [10, 12, 14, 16, 18])
    assert valid.tolist() == [True] * 5


def test_seeded_random_windows_are_deterministic_and_fit():
    kwargs = dict(video_length=200, num_frames=9, target_fps=16, source_fps=30)
    first = temporal_sampling.sample_temporal_window(rng=np.random.default_rng(1234), **kwargs)
    second = temporal_sampling.sample_temporal_window(rng=np.random.default_rng(1234), **kwargs)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[1].all()
    assert first[0][0] >= 0 and first[0][-1] < kwargs['video_length']


def test_short_episode_is_edge_padded_and_marked_invalid():
    indices, valid = temporal_sampling.sample_temporal_window(
        video_length=3,
        num_frames=7,
        target_fps=16,
        source_fps=16,
        random_start=False,
    )

    np.testing.assert_array_equal(indices, [0, 1, 2, 2, 2, 2, 2])
    assert valid.tolist() == [True, True, True, False, False, False, False]
    latent = temporal_sampling.frame_mask_to_latent_mask(valid, temporal_factor=2)
    assert latent.tolist() == [True, True, False, False]


def test_explicit_tail_window_reports_padding_boundary():
    indices, valid = temporal_sampling.sample_temporal_window(
        video_length=20,
        num_frames=5,
        target_fps=10,
        source_fps=10,
        start_frame=18,
    )

    np.testing.assert_array_equal(indices, [18, 19, 19, 19, 19])
    assert valid.tolist() == [True, True, False, False, False]


def test_random_windows_cover_multiple_starts_without_leaking_tail():
    rng = np.random.default_rng(7)
    starts = []
    for _ in range(256):
        indices, valid = temporal_sampling.sample_temporal_window(
            video_length=200,
            num_frames=9,
            target_fps=10,
            source_fps=10,
            rng=rng,
        )
        starts.append(int(indices[0]))
        assert valid.all()
        assert indices[-1] <= 199
    assert min(starts) >= 0
    assert max(starts) <= 191
    assert len(set(starts)) > 32


def test_uniform_compatibility_path_keeps_all_frames_real():
    indices, valid = temporal_sampling.sample_uniform_window(video_length=3, num_frames=7)
    np.testing.assert_array_equal(indices, [0, 0, 0, 1, 1, 1, 2])
    assert valid.all()


def test_masked_edm_loss_matches_original_for_all_valid_mask():
    torch.manual_seed(0)
    target = torch.randn(2, 2, 3, 2, 2)
    prediction = torch.randn_like(target)
    edm = _FakeEDMLoss(target, torch.tensor([[0.5], [2.0]]))
    expected = edm.compute_loss(prediction)
    actual = loss_utils.compute_masked_edm_loss(edm, prediction, torch.ones(2, 3, dtype=torch.bool))
    torch.testing.assert_close(actual, expected)

    shaped = torch.ones(2, 1, 3, 1, 1, dtype=torch.bool)
    torch.testing.assert_close(loss_utils.compute_masked_edm_loss(edm, prediction, shaped), expected)


def test_masked_edm_loss_normalizes_each_sample_and_zero_mask_is_finite():
    target = torch.zeros(2, 1, 4, 1, 1)
    prediction = torch.ones_like(target, requires_grad=True)
    edm = _FakeEDMLoss(target, torch.tensor([[1.0], [3.0]]))
    mask = torch.tensor([[True, False, False, False], [True, True, False, False]])

    loss = loss_utils.compute_masked_edm_loss(edm, prediction, mask)
    torch.testing.assert_close(loss, torch.tensor([1.0, 3.0]))

    zero_loss = loss_utils.compute_masked_edm_loss(edm, prediction, torch.zeros(2, 4, dtype=torch.bool))
    assert torch.isfinite(zero_loss).all()
    assert torch.equal(zero_loss, torch.zeros_like(zero_loss))
    zero_loss.sum().backward()
    assert prediction.grad is not None
    assert torch.equal(prediction.grad, torch.zeros_like(prediction.grad))


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(video_length=0, num_frames=3, target_fps=16),
        dict(video_length=3, num_frames=0, target_fps=16),
        dict(video_length=3, num_frames=3, target_fps=0),
    ],
)
def test_sampling_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        temporal_sampling.sample_temporal_window(**kwargs)


def test_latent_mask_requires_valid_temporal_layout():
    with pytest.raises(ValueError):
        temporal_sampling.frame_mask_to_latent_mask(np.ones(6, dtype=bool), temporal_factor=4)
