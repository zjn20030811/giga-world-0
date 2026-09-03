import copy
import math
import random

import torch
from giga_datasets import video_utils
from giga_train import TRANSFORMS
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from .temporal_sampling import frame_mask_to_latent_mask, sample_temporal_window, sample_uniform_window


@TRANSFORMS.register
class GigaWorld0Transform:
    """Video transformation class for GigaWorld0 training.

    Handles video sampling, resizing, cropping, normalization, and reference frame generation.
    """

    def __init__(
        self,
        num_frames: int,
        height: int,
        width: int,
        image_cfg: dict,
        fps: int = 16,
        sampling_mode: str = 'fixed_fps',
        random_start: bool = True,
        source_fps: float | None = None,
        seed: int | None = None,
    ):
        """Initialize the transform.

        Args:
            num_frames: Number of frames to sample from the video.
            height: Target height for the output frames.
            width: Target width for the output frames.
            image_cfg: Configuration dictionary containing mask generator settings.
            fps: Frames per second for the sampled training window (default: 16).
            sampling_mode: Temporal sampling policy. ``'fixed_fps'`` samples a
                constant-rate window and is the default; ``'uniform'`` keeps
                the original full-episode ``linspace`` behaviour.
            random_start: Randomize the beginning of fixed-rate windows. When
                false, windows start at frame zero.
            source_fps: Optional native video rate. If omitted, the transform
                reads ``video.fps`` or ``data_dict['video_fps']`` when
                available and falls back to ``fps``.
            seed: Optional local seed for temporal, crop, and reference-mask
                randomness.
        """
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps
        self.sampling_mode = sampling_mode.lower()
        if self.sampling_mode not in {'fixed_fps', 'uniform'}:
            raise ValueError("sampling_mode must be either 'fixed_fps' or 'uniform'")
        self.random_start = random_start
        self.source_fps = source_fps
        self.rng = random.Random(seed) if seed is not None else random
        # Normalization transform: convert [0, 1] to [-1, 1]
        self.normalize = transforms.Normalize([0.5], [0.5])
        self.mask_generator = MaskGenerator(**image_cfg['mask_generator'])

    def _get_source_fps(self, video, data_dict: dict) -> float:
        """Resolve the native frame rate used to construct timestamps."""

        candidates = [self.source_fps, data_dict.get('video_fps'), data_dict.get('source_fps')]
        # VideoReaderDecord and VideoReaderCV2 both expose ``fps``. A malformed
        # codec value should not make an otherwise usable sample fail; the
        # configured target rate is a safe one-frame-per-step fallback in that
        # case. Keep this candidate even when an invalid explicit override was
        # supplied so metadata can still recover the sample.
        try:
            candidates.append(video.fps)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                candidate = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate) and candidate > 0:
                return candidate
        return float(self.fps)

    def __call__(self, data_dict):
        """Apply transformations to the input data.

        Args:
            data_dict: Dictionary containing 'video' and 'prompt_embeds'.

        Returns:
            new_data_dict: Transformed data dictionary with processed images and masks.
        """
        video = data_dict['video']
        video_length = len(video)
        source_fps = self._get_source_fps(video, data_dict)
        if self.sampling_mode == 'fixed_fps':
            sample_indexes, frame_valid = sample_temporal_window(
                video_length,
                self.num_frames,
                target_fps=self.fps,
                source_fps=source_fps,
                random_start=self.random_start,
                rng=self.rng,
            )
        else:
            sample_indexes, frame_valid = sample_uniform_window(video_length, self.num_frames)
        # Fixed-rate windows are usually far shorter than an episode. Use the
        # reader's direct batched path so a late random window does not decode
        # every preceding frame; retain method 2 for the legacy policy.
        strictly_increasing = bool((sample_indexes[1:] > sample_indexes[:-1]).all())
        sample_method = 1 if self.sampling_mode == 'fixed_fps' and strictly_increasing else 2
        input_images = video_utils.sample_video(video, sample_indexes, method=sample_method)
        # Convert to tensor and rearrange dimensions: (T, H, W, C) -> (T, C, H, W)
        input_images = torch.from_numpy(input_images).permute(0, 3, 1, 2).contiguous()

        image_height = input_images.shape[2]
        image_width = input_images.shape[3]
        dst_width, dst_height = self.width, self.height

        # Calculate new dimensions maintaining aspect ratio
        if float(dst_height) / image_height < float(dst_width) / image_width:
            new_height = int(round(float(dst_width) / image_width * image_height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / image_height * image_width))

        # Random crop coordinates
        x1 = self.rng.randint(0, new_width - dst_width)
        y1 = self.rng.randint(0, new_height - dst_height)

        # Apply resize and crop
        input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
        input_images = F.crop(input_images, y1, x1, dst_height, dst_width)

        # ===== Normalize =====
        # Scale to [0, 1]
        input_images = input_images / 255.0
        # Normalize to [-1, 1]
        input_images = self.normalize(input_images)

        # ===== Generate Reference Images and Masks =====
        # Get masks for reference frames
        ref_masks, ref_latent_masks = self.mask_generator.get_mask(input_images.shape[0], rng=self.rng)
        # Expand dimensions for broadcasting: (T,) -> (T, 1, 1, 1)
        ref_masks = ref_masks[:, None, None, None]
        # Expand for latent space: (T_latent,) -> (1, T_latent, 1, 1)
        ref_latent_masks = ref_latent_masks[None, :, None, None]
        # Create reference images by masking
        ref_images = copy.deepcopy(input_images)
        ref_images = ref_images * ref_masks

        # The VAE has one temporal anchor latent followed by groups of frames.
        # Keep both masks: ``frame_mask`` is useful to data consumers, while
        # ``latent_mask`` lets the trainer exclude padded tail latents from the
        # diffusion objective.
        frame_mask = torch.from_numpy(frame_valid)
        latent_mask = torch.from_numpy(frame_mask_to_latent_mask(frame_valid, self.mask_generator.factor))

        new_data_dict = dict(
            fps=self.fps,
            images=input_images,
            ref_images=ref_images,
            ref_masks=ref_latent_masks,
            frame_mask=frame_mask,
            latent_mask=latent_mask,
            sample_indices=torch.from_numpy(sample_indexes),
            prompt_embeds=data_dict['prompt_embeds'],
        )
        return new_data_dict


class MaskGenerator:
    """Generates binary masks for reference frames in video sequences.

    Used to control which frames are treated as reference (conditioning) frames during training.
    """

    def __init__(self, max_ref_frames: int, factor: int = 8, start: int = 1):
        """Initialize the mask generator.

        Args:
            max_ref_frames: Maximum number of reference frames (must satisfy: (max_ref_frames - 1) % factor == 0).
            factor: Downsampling factor between frame space and latent space (default: 8).
            start: Minimum number of reference latents to generate (default: 1).
        """
        assert max_ref_frames > 0 and (max_ref_frames - 1) % factor == 0
        self.max_ref_frames = max_ref_frames
        self.factor = factor
        self.start = start
        # Calculate maximum reference latents based on factor
        self.max_ref_latents = 1 + (max_ref_frames - 1) // factor
        assert self.start <= self.max_ref_latents

    def get_mask(self, num_frames: int, rng=None):
        """Generate binary masks for reference frames and latents.

        Args:
            num_frames: Total number of frames in the sequence.

        Returns:
            ref_masks: Binary mask tensor for frame space (shape: (num_frames,)).
                      1.0 for reference frames, 0.0 for non-reference frames.
            ref_latent_masks: Binary mask tensor for latent space (shape: (num_latents,)).
                             1.0 for reference latents, 0.0 for non-reference latents.
        """
        # Validate input dimensions
        assert num_frames > 0 and (num_frames - 1) % self.factor == 0 and num_frames >= self.max_ref_frames

        # Calculate number of latents based on downsampling factor
        num_latents = 1 + (num_frames - 1) // self.factor

        # Randomly select number of reference latents. Accepting an injected
        # RNG keeps seeded transforms reproducible without changing callers
        # that rely on the module-level random generator.
        rng = random if rng is None else rng
        if hasattr(rng, 'integers'):
            num_ref_latents = int(rng.integers(self.start, self.max_ref_latents + 1))
        else:
            num_ref_latents = rng.randint(self.start, self.max_ref_latents)

        # Calculate corresponding number of reference frames
        if num_ref_latents > 0:
            num_ref_frames = 1 + (num_ref_latents - 1) * self.factor
        else:
            num_ref_frames = 0

        # Create binary mask for frames
        ref_masks = torch.zeros((num_frames,), dtype=torch.float32)
        ref_masks[:num_ref_frames] = 1  # Mark first N frames as reference

        # Create binary mask for latents
        ref_latent_masks = torch.zeros((num_latents,), dtype=torch.float32)
        ref_latent_masks[:num_ref_latents] = 1  # Mark first N latents as reference

        return ref_masks, ref_latent_masks
