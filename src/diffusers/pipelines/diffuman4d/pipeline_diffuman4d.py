# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy
from tqdm import tqdm

import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin

from ...models.unets.unet_multiview_condition import UNetMultiviewConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import Diffuman4DPipeline
        ```
"""


def _get_time_dim(x: torch.Tensor) -> int:
    if x.ndim == 4:
        return 0
    if x.ndim == 5:
        return 1
    if x.ndim == 6:
        return 2
    raise ValueError(f"Unsupported ndim={x.ndim}. Expected 4D/5D/6D tensor.")


def _get_time_length(x: torch.Tensor) -> int:
    return x.shape[_get_time_dim(x)]


def _get_wan_t_target(t: int) -> int:
    # Wan VAE temporal rule: T_latent = 1 + (T - 1) // 4.
    # To avoid tail truncation, pad T to: 1 + 4 * ceil((T - 1) / 4).
    if t < 1:
        raise ValueError(f"`T` must be >= 1, got {t}.")
    return 1 + ((t + 2) // 4) * 4


def _pad_time_repeat_last(x: Optional[torch.Tensor], t_target: int) -> Optional[torch.Tensor]:
    if x is None:
        return None
    t_dim = _get_time_dim(x)
    t = x.shape[t_dim]
    if t >= t_target:
        return x

    pad = t_target - t
    last = x.select(dim=t_dim, index=t - 1).unsqueeze(t_dim)
    repeats = [1] * x.ndim
    repeats[t_dim] = pad
    pad_tensor = last.repeat(*repeats)
    return torch.cat([x, pad_tensor], dim=t_dim)


def _select_wan_group_first_temporal(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Compress temporal axis with Wan grouping rule:
    groups are [0], [1:5], [5:9], ... and we take the first frame of each group.

    Example indices: [0, 1, 5, 9, ...]
    """
    if x is None:
        return None
    t_dim = _get_time_dim(x)
    t = x.shape[t_dim]
    # [0] + [1 + 4k]
    idx = [0]
    cur = 1
    while cur < t:
        idx.append(cur)
        cur += 4
    index = torch.tensor(idx, device=x.device, dtype=torch.long)
    return x.index_select(dim=t_dim, index=index)


def _flatten_to_frame_batch(
    x: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[int], Optional[int], Optional[int]]:
    """
    Convert condition layouts to frame-batch layout expected by current 2D UNet path.

    Returns:
    - flat tensor: [N, C, H, W]
    - batch_size: number of scenes (B)
    - num_views: number of views (V)
    - frames_per_view: temporal length per view (T')
    """
    if x is None:
        return None, None, None, None

    if x.ndim == 4:
        t, c, h, w = x.shape
        return x, 1, 1, t
    if x.ndim == 5:
        b, t, c, h, w = x.shape
        return x.reshape(b * t, c, h, w), b, 1, t
    if x.ndim == 6:
        b, v, t, c, h, w = x.shape
        return x.reshape(b * v * t, c, h, w), b, v, t
    raise ValueError(f"Unsupported ndim={x.ndim}. Expected 4D/5D/6D tensor.")


def encode_vae(vae, pixel_values):
    """
    Encode one temporal sequence with VAE.

    Expected input shape: [T, C, H, W].
    For Wan VAE, T is the temporal axis and should be encoded together.
    """
    if pixel_values.ndim != 4:
        raise ValueError(
            f"`pixel_values` must be 4D [T,C,H,W], got shape {tuple(pixel_values.shape)}."
        )

    # Prefer Wan-specific unified API when available.
    if hasattr(vae, "encode_for_unet"):
        return vae.encode_for_unet(pixel_values)

    # Fallback: standard AutoencoderKL-style API.
    # Keep full temporal sequence in one pass to preserve continuity.
    out = vae.encode(pixel_values).latent_dist.sample()
    return out * vae.config.scaling_factor


def decode_vae(vae, latents, generator=None):
    # Do NOT split on temporal axis; decode the whole sequence to keep consistency.
    return vae.decode(
        latents / vae.config.scaling_factor,
        return_dict=False,
        generator=generator,
    )[0]


def encode_image_vae(vae, images, image_latents=None, dtype=None, device=None):
    """
    Encode video batches with Wan VAE.

    Supported inputs:
    - [B, T, C, H, W]
    - [B, V, T, C, H, W]

    Returns:
    - [B, T', Z, H', W']
    - [B, V, T', Z, H', W']
    """
    dtype = vae.dtype if dtype is None else dtype
    device = vae.device if device is None else device

    if image_latents is None:
        if images is None:
            return None
        images = images.to(dtype=dtype, device=device)
        if images.ndim == 5:
            # [B, T, C, H, W] -> encode each sample independently.
            latents_b = [encode_vae(vae, images[b]) for b in range(images.shape[0])]
            image_latents = torch.stack(latents_b, dim=0)
        elif images.ndim == 6:
            # [B, V, T, C, H, W] -> encode each (scene, view) independently.
            latents_bv = []
            for b in range(images.shape[0]):
                latents_v = [encode_vae(vae, images[b, v]) for v in range(images.shape[1])]
                latents_bv.append(torch.stack(latents_v, dim=0))
            image_latents = torch.stack(latents_bv, dim=0)
        else:
            raise ValueError(
                f"`images` must be 5D [B,T,C,H,W] or 6D [B,V,T,C,H,W], got shape {tuple(images.shape)}."
            )
    else:
        image_latents = image_latents.to(dtype=dtype, device=device)
        if image_latents.ndim not in (5, 6):
            raise ValueError(
                f"`image_latents` must be 5D [B,T,C,H,W] or 6D [B,V,T,C,H,W], got shape {tuple(image_latents.shape)}."
            )

    return image_latents


def encode_image_resizing(images, image_latents=None, shape=None, mode="bilinear", dtype=None, device=None):
    """
    Resize non-VAE conditions to latent resolution while preserving layout.

    Supported inputs:
    - [T, C, H, W]
    - [B, T, C, H, W]
    - [B, V, T, C, H, W]

    `shape` is expected to be `(T', H', W')` so conditions align with Wan latent.
    """
    if shape is None or len(shape) != 3:
        raise ValueError(f"`shape` must be (T',H',W'), got {shape}.")

    def _interpolate_3d(x: torch.Tensor, out_shape: tuple[int, int, int], interp_mode: str) -> torch.Tensor:
        mode_3d = "trilinear" if interp_mode == "bilinear" else interp_mode
        kwargs = {"size": out_shape, "mode": mode_3d}
        if mode_3d in ("trilinear",):
            kwargs["align_corners"] = False
        return torch.nn.functional.interpolate(x, **kwargs)

    if image_latents is None:
        if images is None:
            return None

        if dtype is not None:
            images = images.to(dtype=dtype)
        if device is not None:
            images = images.to(device=device)

        if images.ndim == 4:
            # [T,C,H,W] -> [1,C,T,H,W] -> resize -> [T',C,H',W']
            x = images.permute(1, 0, 2, 3).unsqueeze(0)
            x = _interpolate_3d(x, out_shape=shape, interp_mode=mode)
            image_latents = x.squeeze(0).permute(1, 0, 2, 3).contiguous()
        elif images.ndim == 5:
            # [B,T,C,H,W] -> [B,C,T,H,W] -> resize -> [B,T',C,H',W']
            x = images.permute(0, 2, 1, 3, 4)
            x = _interpolate_3d(x, out_shape=shape, interp_mode=mode)
            image_latents = x.permute(0, 2, 1, 3, 4).contiguous()
        elif images.ndim == 6:
            # [B,V,T,C,H,W] -> [B*V,C,T,H,W] -> resize -> [B,V,T',C,H',W']
            b, v, t, c, h, w = images.shape
            x = images.permute(0, 1, 3, 2, 4, 5).reshape(b * v, c, t, h, w)
            x = _interpolate_3d(x, out_shape=shape, interp_mode=mode)
            t_new, h_new, w_new = shape
            image_latents = x.reshape(b, v, c, t_new, h_new, w_new).permute(0, 1, 3, 2, 4, 5).contiguous()
        else:
            raise ValueError(
                f"`images` must be 4D [T,C,H,W], 5D [B,T,C,H,W], or 6D [B,V,T,C,H,W], got {tuple(images.shape)}."
            )

    if dtype is not None:
        image_latents = image_latents.to(dtype=dtype)
    if device is not None:
        image_latents = image_latents.to(device=device)
    return image_latents


def get_negative_latents(latents, color="black"):
    negative_latents = torch.ones_like(latents)
    if color == "black":
        negative_latents = -1.0 * negative_latents
    elif color == "white":
        negative_latents = negative_latents
    elif color == "grey" or color == "random":
        negative_latents = 0.0 * negative_latents
    else:
        raise ValueError(f"color: {color} not supported.")
    return negative_latents


class Diffuman4DPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for Diffuman4D.
    """

    model_cpu_offload_seq = "unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNetMultiviewConditionModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_all_latents(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        plucker_embeds: Optional[torch.Tensor] = None,
        skeletons: Optional[torch.Tensor] = None,
        cond_masks: Optional[torch.Tensor] = None,
        pixel_values_latents: Optional[torch.Tensor] = None,
        plucker_embeds_latents: Optional[torch.Tensor] = None,
        skeletons_latents: Optional[torch.Tensor] = None,
        cond_masks_latents: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        # image
        dtype, device = self.vae.dtype, self._execution_device
        if pixel_values is None and pixel_values_latents is None:
            raise ValueError("`pixel_values` or `pixel_values_latents` must be provided.")

        # Raw frames need Wan temporal padding before VAE encoding.
        # Pre-encoded latents should preserve their existing temporal length.
        if pixel_values is not None:
            t_target = _get_wan_t_target(_get_time_length(pixel_values))
        else:
            t_target = _get_time_length(pixel_values_latents)

        # Keep all branches temporally aligned before encoding/resizing.
        pixel_values = _pad_time_repeat_last(pixel_values, t_target)
        plucker_embeds = _pad_time_repeat_last(plucker_embeds, t_target)
        skeletons = _pad_time_repeat_last(skeletons, t_target)
        cond_masks = _pad_time_repeat_last(cond_masks, t_target)
        pixel_values_latents = _pad_time_repeat_last(pixel_values_latents, t_target)
        plucker_embeds_latents = _pad_time_repeat_last(plucker_embeds_latents, t_target)
        skeletons_latents = _pad_time_repeat_last(skeletons_latents, t_target)
        cond_masks_latents = _pad_time_repeat_last(cond_masks_latents, t_target)

        pixel_values_latents = encode_image_vae(
            self.vae,
            images=pixel_values,
            image_latents=pixel_values_latents,
            dtype=dtype,
            device=device,
        )
        if pixel_values_latents.ndim == 4:
            t_latent, latent_dim, height, width = pixel_values_latents.shape
        elif pixel_values_latents.ndim == 5:
            t_latent, latent_dim, height, width = pixel_values_latents.shape[1:]
        elif pixel_values_latents.ndim == 6:
            t_latent, latent_dim, height, width = pixel_values_latents.shape[2:]
        else:
            raise ValueError(
                f"`pixel_values_latents` must be 4D/5D/6D, got shape {tuple(pixel_values_latents.shape)}."
            )
        latent_shape = (t_latent, height, width)

        # plucker
        plucker_embeds_latents = encode_image_resizing(
            images=plucker_embeds,
            image_latents=plucker_embeds_latents,
            shape=latent_shape,
            mode="bilinear",
            dtype=dtype,
            device=device,
        )

        # skeletons
        if skeletons_latents is not None:
            skeletons_latents = skeletons_latents.to(dtype=dtype, device=device)
        elif self.unet.config.enable_pose_encoder:
            if skeletons is None:
                raise ValueError(
                    "`skeletons` or `skeletons_latents` must be provided when pose encoder is enabled."
                )
            skeletons_latents = skeletons.to(dtype=dtype, device=device)
        else:
            skeletons_latents = encode_image_vae(
                self.vae,
                images=skeletons,
                image_latents=skeletons_latents,
                dtype=dtype,
                device=device,
            )

        # conditional masks
        if cond_masks is None and cond_masks_latents is None:
            raise ValueError("`cond_masks` or `cond_masks_latents` must be provided.")
        cond_masks_latents = encode_image_resizing(
            images=cond_masks,
            image_latents=cond_masks_latents,
            shape=latent_shape,
            mode="nearest",
            dtype=dtype,
            device=device,
        )
        # Keep masks temporally aligned by repeating edge frames.
        cond_masks_latents = _pad_time_repeat_last(cond_masks_latents, t_latent)

        pixel_values_latents, batch_size, num_views, frames_per_view = _flatten_to_frame_batch(pixel_values_latents)
        plucker_embeds_latents, p_batch, p_views, p_frames = _flatten_to_frame_batch(plucker_embeds_latents)
        cond_masks_latents, m_batch, m_views, m_frames = _flatten_to_frame_batch(cond_masks_latents)

        for name, b, v, t in (
            ("plucker_embeds_latents", p_batch, p_views, p_frames),
            ("cond_masks_latents", m_batch, m_views, m_frames),
        ):
            if b is not None and (b != batch_size or v != num_views or t != frames_per_view):
                raise ValueError(
                    f"{name} shape mismatch: expected (B,V,T)=({batch_size},{num_views},{frames_per_view}), got ({b},{v},{t})."
                )

        if self.unet.config.enable_pose_encoder:
            # Keep 5D/6D skeleton layout for 3D pose encoder inside UNet.
            if skeletons_latents is not None and skeletons_latents.ndim not in (5, 6):
                raise ValueError(
                    f"`skeletons_latents` must be 5D/6D when pose encoder is enabled, got {tuple(skeletons_latents.shape)}."
                )
            s_batch = s_views = s_frames = None
            if skeletons_latents is not None:
                if skeletons_latents.ndim == 5:
                    s_batch, s_views, s_frames = skeletons_latents.shape[0], 1, skeletons_latents.shape[1]
                else:
                    s_batch, s_views, s_frames = skeletons_latents.shape[0], skeletons_latents.shape[1], skeletons_latents.shape[2]
                if (s_batch, s_views, s_frames) != (batch_size, num_views, frames_per_view):
                    raise ValueError(
                        f"skeletons_latents shape mismatch: expected (B,V,T)=({batch_size},{num_views},{frames_per_view}), got ({s_batch},{s_views},{s_frames})."
                    )
        else:
            skeletons_latents, s_batch, s_views, s_frames = _flatten_to_frame_batch(skeletons_latents)
            if s_batch is not None and (s_batch != batch_size or s_views != num_views or s_frames != frames_per_view):
                raise ValueError(
                    f"skeletons_latents shape mismatch: expected (B,V,T)=({batch_size},{num_views},{frames_per_view}), got ({s_batch},{s_views},{s_frames})."
                )

        # latents
        latents = self.prepare_latents(
            batch_size * num_views * frames_per_view,
            latent_dim,
            height * self.vae_scale_factor,
            width * self.vae_scale_factor,
            dtype,
            device,
            generator,
            latents,
        )

        # batch_size: scene batch B; num_views: view count V; frames_per_view: latent temporal length T'
        return (
            pixel_values_latents,
            plucker_embeds_latents,
            skeletons_latents,
            cond_masks_latents,
            latents,
            batch_size,
            num_views,
            frames_per_view,
        )

    def parepare_schedulers(self, num_inference_steps: int, num_frames: int):
        # create a scheduler for each latent
        device = self._execution_device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        schedulers = [deepcopy(self.scheduler) for _ in range(num_frames)]
        timesteps = self.scheduler.timesteps
        return schedulers, timesteps

    def get_timestep(self, timesteps, timestep_indices, is_cond):
        # timestep of conditional latents === 0
        timestep_indices[is_cond] = 0
        timestep = timesteps[timestep_indices]
        timestep[is_cond] = 0
        return timestep

    def post_process(
        self,
        latents,
        output_type="pt",
        generator=None,
        batch_size: int = 1,
        num_views: int = 1,
        frames_per_view: Optional[int] = None,
    ):
        if frames_per_view is None:
            images = decode_vae(self.vae, latents, generator=generator)
        else:
            latent_dim, h, w = latents.shape[1:]
            expected = batch_size * num_views * frames_per_view
            if latents.shape[0] != expected:
                raise ValueError(
                    f"Latent batch mismatch: expected N={expected} from B={batch_size}, V={num_views}, T={frames_per_view}, got {latents.shape[0]}."
                )
            latents = latents.reshape(batch_size, num_views, frames_per_view, latent_dim, h, w)
            decoded = []
            for b in range(batch_size):
                for v in range(num_views):
                    out = decode_vae(self.vae, latents[b, v], generator=generator)
                    decoded.append(out)
            images = torch.stack(decoded, dim=0).reshape(-1, *decoded[0].shape[1:])
        images = self.image_processor.postprocess(
            images, output_type=output_type, do_denormalize=[True] * images.shape[0]
        )
        return images

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        plucker_embeds: Optional[torch.Tensor] = None,
        skeletons: Optional[torch.Tensor] = None,
        cond_masks: Optional[torch.Tensor] = None,
        pixel_values_latents: Optional[torch.Tensor] = None,
        plucker_embeds_latents: Optional[torch.Tensor] = None,
        skeletons_latents: Optional[torch.Tensor] = None,
        cond_masks_latents: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        domains: List[str] = None,
        num_inference_steps: int = 1,
        schedulers: Optional[List[object]] = None,
        timesteps: Optional[List[int]] = None,
        timestep_indices: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "latent",
    ):
        r"""
        Denoise a sample sequence for num_inference_steps starting from timestep_indices.

        Shape conventions:
        - single-view: [B, T, C, H, W]
        - multi-view:  [B, V, T, C, H, W]
        Internally flattened to [B*V*T, C, H, W] for the 2D UNet backbone,
        while `num_views` and `num_frames` are passed to attention blocks.

        Examples:
        """

        # Deafult args
        dtype = self.vae.dtype
        device = self._execution_device

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # Prepare conditions and latents
        # batch_size = B, num_views = V, frames_per_view = T' (Wan-latent temporal length)
        pixel_values_latents, plucker_embeds_latents, skeletons_latents, cond_masks_latents, latents, batch_size, num_views, frames_per_view = (
            self.prepare_all_latents(
                pixel_values=pixel_values,
                plucker_embeds=plucker_embeds,
                skeletons=skeletons,
                cond_masks=cond_masks,
                pixel_values_latents=pixel_values_latents,
                plucker_embeds_latents=plucker_embeds_latents,
                skeletons_latents=skeletons_latents,
                cond_masks_latents=cond_masks_latents,
                latents=latents,
                generator=generator,
            )
        )
        if domains is None:
            domains = ["spatial"] * batch_size
        elif len(domains) == 1 and batch_size > 1:
            domains = domains * batch_size
        elif len(domains) != batch_size:
            raise ValueError(
                f"`domains` length must match batch size ({batch_size}), got {len(domains)}."
            )

        is_cond = cond_masks_latents[:, 0, 0, 0] == 0

        # Concatenate the unconditional and conditional embeddings into a single batch
        if self.do_classifier_free_guidance:
            negative_pixel_values_latents = get_negative_latents(pixel_values_latents, color="white")
            if plucker_embeds_latents is not None:
                negative_plucker_embeds_latents = get_negative_latents(plucker_embeds_latents, color="grey")
                plucker_embeds_latents = torch.cat([negative_plucker_embeds_latents, plucker_embeds_latents])
            if skeletons_latents is not None:
                negative_skeletons_latents = get_negative_latents(skeletons_latents, color="black")
                skeletons_latents = torch.cat([negative_skeletons_latents, skeletons_latents])
            cond_masks_latents = torch.cat([cond_masks_latents] * 2)
            domains = domains * 2

        # Prepare timesteps
        # denoise num_inference_steps starting from timestep_indices
        if schedulers is None:
            schedulers, timesteps = self.parepare_schedulers(num_inference_steps, len(latents))
        if timesteps is None:
            raise ValueError("`timesteps` must be provided when `schedulers` is provided.")
        if len(schedulers) != len(latents):
            raise ValueError(
                f"`schedulers` length must match latent batch ({len(latents)}), got {len(schedulers)}."
            )
        if timestep_indices is None:
            timestep_indices = torch.zeros(len(latents), device=device, dtype=torch.long)
        else:
            timestep_indices = timestep_indices.to(device=device, dtype=torch.long)
            if timestep_indices.numel() != len(latents):
                raise ValueError(
                    f"`timestep_indices` size mismatch: expected {len(latents)}, got {timestep_indices.numel()}."
                )

        # Denoising loop
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for _ in range(num_inference_steps):
                if self.interrupt:
                    continue

                timestep = self.get_timestep(timesteps, timestep_indices, is_cond)

                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

                # Replace the noise latents with the conditional image latents
                latent_model_input[is_cond, ...] = pixel_values_latents[is_cond, ...]

                # expand the latents if we are doing classifier free guidance
                if self.do_classifier_free_guidance:
                    timestep = torch.cat([timestep] * 2)
                    negative_latent_model_input = latent_model_input.clone()
                    negative_latent_model_input[is_cond, ...] = negative_pixel_values_latents[is_cond, ...]
                    latent_model_input = torch.cat([negative_latent_model_input, latent_model_input])

                # Concat the latents along the channel dimension
                latent_model_input = [latent_model_input]
                if plucker_embeds_latents is not None:
                    latent_model_input.append(plucker_embeds_latents)
                if skeletons_latents is not None and not self.unet.config.enable_pose_encoder:
                    latent_model_input.append(skeletons_latents)
                latent_model_input.append(cond_masks_latents)
                latent_model_input = torch.cat(latent_model_input, dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    timestep=timestep,
                    skeletons=skeletons_latents,
                    domains=domains,
                    num_frames=frames_per_view,
                    num_views=num_views,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                new_latents = []
                for j in range(len(noise_pred)):
                    t = timestep[j].item()
                    noise = noise_pred[j].unsqueeze(0)
                    latent = latents[j].unsqueeze(0)
                    # only denoise target latents
                    if not is_cond[j]:
                        latent = schedulers[j].step(noise, t, latent, return_dict=False)[0]
                    new_latents.append(latent.to(dtype=dtype))
                latents = torch.cat(new_latents)
                timestep_indices[~is_cond] += 1

                progress_bar.update()

        # Postprocess the denoised latents
        if output_type != "latent":
            # output_type: pt, np, pil
            images = self.post_process(
                latents,
                output_type,
                generator,
                batch_size=batch_size,
                num_views=num_views,
                frames_per_view=frames_per_view,
            )
        else:
            images = latents

        # Offload all models
        self.maybe_free_model_hooks()

        return images

    def sliding_iterative_denoise(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        plucker_embeds: Optional[torch.Tensor] = None,
        skeletons: Optional[torch.Tensor] = None,
        cond_masks: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        domain: str = "spatial",  # "spatial" or "temporal"
        timestep_indices: Optional[torch.Tensor] = None,
        # sliding denoising args
        window_size: int = 12,
        sliding_stride: int = 1,
        sliding_shift: int = 0,
        bidirectional: bool = True,
        num_denoising_steps: int = 1,
        alternation_rounds: int = 3,
        guidance_scale: float = 2.0,
        tqdm: Callable = tqdm,
    ):
        """
        Denoise a spatial or temporal sample sequence with sliding iterative denoising scheme.
        """
        dtype, device = self.vae.dtype, self._execution_device

        if (window_size * num_denoising_steps) % sliding_stride != 0:
            raise ValueError(
                f"The window size ({window_size}) * num denoising steps ({num_denoising_steps}) "
                f"should be divisible by the sliding stride ({sliding_stride})"
            )
        num_denoising_steps_peralt = window_size * num_denoising_steps // sliding_stride
        if bidirectional:
            num_denoising_steps_peralt *= 2
        # num_inference_steps is the total number of denoising steps for each sample
        num_inference_steps = num_denoising_steps_peralt * alternation_rounds

        # prepare input latents
        (
            pixel_values_latents,
            plucker_embeds_latents,
            skeletons_latents,
            cond_masks_latents,
            latents,
            batch_size,
            num_views,
            frames_per_view,
        ) = (
            self.prepare_all_latents(
                pixel_values=pixel_values,
                plucker_embeds=plucker_embeds,
                skeletons=skeletons,
                cond_masks=cond_masks,
                latents=latents,
            )
        )

        n_total = batch_size * num_views * frames_per_view
        def unflatten_frames(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is None or x.ndim != 4:
                return x
            c, h, w = x.shape[1:]
            return x.reshape(batch_size, num_views, frames_per_view, c, h, w)

        # Keep grouped tensors for window slicing and re-flatten inside `self(...)`.
        pixel_values_latents_grouped = unflatten_frames(pixel_values_latents)
        plucker_embeds_latents_grouped = unflatten_frames(plucker_embeds_latents)
        cond_masks_latents_grouped = unflatten_frames(cond_masks_latents)
        if skeletons_latents is not None:
            if skeletons_latents.ndim == 4:
                skeletons_latents_grouped = unflatten_frames(skeletons_latents)
            elif skeletons_latents.ndim == 5:
                skeletons_latents_grouped = skeletons_latents.unsqueeze(1)
            else:
                skeletons_latents_grouped = skeletons_latents
        else:
            skeletons_latents_grouped = None

        if timestep_indices is None:
            timestep_indices = torch.zeros(n_total, device=device, dtype=torch.long)
        else:
            timestep_indices = timestep_indices.to(device=device, dtype=torch.long)
            if timestep_indices.numel() != n_total:
                raise ValueError(
                    f"`timestep_indices` size mismatch: expected {n_total} (B*V*T), got {timestep_indices.numel()}."
                )

        # Build time-level masks from flattened condition masks.
        cond_mask_flat = cond_masks_latents[:, 0, 0, 0].reshape(batch_size, num_views, frames_per_view)
        cond_mask_ref = cond_mask_flat[0, 0]
        if not torch.all(cond_mask_flat == cond_mask_ref.view(1, 1, -1)):
            raise ValueError(
                "All views/samples must share the same temporal cond-mask pattern for sliding denoising."
            )
        input_times = torch.where(cond_mask_ref == 0.0)[0].to(device=device)
        target_times = torch.where(cond_mask_ref != 0.0)[0].to(device=device)
        if target_times.numel() == 0:
            raise ValueError("No target frames found for denoising.")

        target_timestep_indices = timestep_indices.reshape(batch_size, num_views, frames_per_view)[:, :, target_times].reshape(-1)
        input_timestep_indices = timestep_indices.reshape(batch_size, num_views, frames_per_view)[:, :, input_times].reshape(-1)
        target_timestep_start = target_timestep_indices[0].item()
        if (target_timestep_indices != target_timestep_indices[0]).any():
            raise ValueError(
                f"The timestep indices should be the same for all target samples, timestep_indices = {timestep_indices}"
            )
        if input_timestep_indices.numel() > 0 and (input_timestep_indices != 0).any():
            raise ValueError(
                f"The timestep indices should be 0 for all input samples, timestep_indices = {timestep_indices}"
            )

        # prepare schedulers
        schedulers, timesteps = self.parepare_schedulers(num_inference_steps, len(latents))

        def gather_frame_indices(time_ids: torch.Tensor) -> torch.Tensor:
            # Map (B,V,T) -> flattened frame indices in [B*V*T].
            ids = []
            for b in range(batch_size):
                for v in range(num_views):
                    base = (b * num_views + v) * frames_per_view
                    ids.append(base + time_ids)
            return torch.cat(ids, dim=0)

        if domain not in ("spatial", "temporal"):
            raise ValueError(f"Unsupported domain: {domain}")

        # For temporal domain we follow the paper's half-half assumption:
        # [cond_half, target_half] with one-to-one temporal pairing.
        if domain == "temporal":
            n_input = input_times.numel()
            if n_input == 0:
                raise ValueError("Temporal sliding requires non-empty condition half.")
            if target_times.numel() != n_input:
                raise ValueError(
                    "Temporal sliding expects half-half layout: number of target times must equal number of input times."
                )
            paired_targets = input_times + n_input
            if not torch.equal(target_times, paired_targets):
                raise ValueError(
                    "Temporal sliding expects ordered half-half times: target_times must equal input_times + len(input_times)."
                )

        # prepare sliding windows on target times
        target_time_windows = []
        directions = (-1, 1) if bidirectional else (-1,)
        for direction in directions:
            for shift in range(sliding_shift, sliding_shift + len(target_times), sliding_stride):
                target_window = target_times.roll(shifts=shift * direction)[:window_size]
                target_time_windows.append(target_window)

        # Count how many denoising updates each target time receives.
        target_update_counts = torch.zeros(frames_per_view, device=device, dtype=torch.long)
        for target_window in target_time_windows:
            target_update_counts[target_window] += 1

        # sliding iterative denoising
        for target_window in tqdm(target_time_windows, total=len(target_time_windows)):
            if domain == "spatial":
                # Spatial stage: keep all condition times and slide target times.
                input_window = input_times
            else:
                # Temporal stage: pair each target with its condition counterpart from the first half.
                input_window = target_window - input_times.numel()

            # Keep [cond_half, target_half] order for temporal embedding assumptions.
            time_window = torch.cat([input_window, target_window])

            frame_window = gather_frame_indices(time_window)
            target_frame_window = gather_frame_indices(target_window)
            frame_window_list = frame_window.detach().cpu().tolist()

            def get_slice(x):
                if x is None:
                    return None
                if x.ndim == 4:
                    return x[frame_window]
                if x.ndim == 5:
                    return x[:, time_window]
                if x.ndim == 6:
                    return x[:, :, time_window]
                raise ValueError(f"Unsupported ndim for slicing: {x.ndim}")

            # few-step denoising for each window
            latents_window = self(
                pixel_values_latents=get_slice(pixel_values_latents_grouped),
                plucker_embeds_latents=get_slice(plucker_embeds_latents_grouped),
                skeletons_latents=get_slice(skeletons_latents_grouped),
                cond_masks_latents=get_slice(cond_masks_latents_grouped),
                latents=get_slice(latents),
                domains=[domain] * batch_size,
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                schedulers=[schedulers[i] for i in frame_window_list],
                timesteps=timesteps,
                timestep_indices=timestep_indices[frame_window],
                output_type="latent",
            )

            # update latents and timesteps
            timestep_indices[target_frame_window] += num_denoising_steps
            latents[frame_window] = latents_window

        # sanity check
        target_indices = gather_frame_indices(target_times)
        input_indices = gather_frame_indices(input_times)
        expected_target_steps = target_timestep_start + target_update_counts[target_times] * num_denoising_steps
        expected_target_steps = expected_target_steps.reshape(1, 1, -1).expand(batch_size, num_views, -1).reshape(-1)
        actual_target_steps = timestep_indices[target_indices]
        if (actual_target_steps != expected_target_steps).any():
            raise ValueError(
                "The denoised timesteps of target samples mismatch the config, "
                f"expected={expected_target_steps}, actual={actual_target_steps}"
            )
        if input_indices.numel() > 0 and (timestep_indices[input_indices] != 0).any():
            raise ValueError(f"Timesteps of input samples have changed, timestep_indices = {timestep_indices}")

        images = self.post_process(
            latents,
            output_type="pt",
            batch_size=batch_size,
            num_views=num_views,
            frames_per_view=frames_per_view,
        )
        return {
            "images": images,
            "latents": latents,
            "timestep_indices": timestep_indices,
            "fully_denoised": timestep_indices == num_inference_steps,
        }
