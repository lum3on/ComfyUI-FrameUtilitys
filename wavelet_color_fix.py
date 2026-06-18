"""
Wavelet color correction node for ComfyUI Frame Utilitys.

Color correction helpers are adapted from:
https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py
"""

import logging
from typing import Tuple

import comfy.utils
import torch
import torch.nn.functional as F


def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(feat.size()) != 4:
        raise ValueError("Expected a 4D tensor in [B, C, H, W] format")

    batch, channels = feat.size()[:2]
    feat_var = feat.reshape(batch, channels, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(batch, channels, 1, 1)
    feat_mean = feat.reshape(batch, channels, -1).mean(dim=2).reshape(batch, channels, 1, 1)
    return feat_mean, feat_std


def _adaptive_instance_normalization(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    size = content.size()
    style_mean, style_std = _calc_mean_std(style)
    content_mean, content_std = _calc_mean_std(content)
    normalized = (content - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def _wavelet_blur(image: torch.Tensor, radius: int) -> torch.Tensor:
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    channels = image.shape[1]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(channels, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(image, kernel, groups=channels, dilation=radius)


def _wavelet_decomposition(image: torch.Tensor, levels: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    high_freq = torch.zeros_like(image)
    low_freq = image

    for i in range(levels):
        radius = 2 ** i
        low_freq = _wavelet_blur(image, radius)
        high_freq += image - low_freq
        image = low_freq

    return high_freq, low_freq


def _wavelet_reconstruction(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    content_high_freq, _ = _wavelet_decomposition(content)
    _, style_low_freq = _wavelet_decomposition(style)
    return content_high_freq + style_low_freq


def _resize_source_to_target(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[1:3] == target.shape[1:3]:
        return source

    source_nchw = source.permute(0, 3, 1, 2)
    resized = F.interpolate(
        source_nchw,
        size=(target.shape[1], target.shape[2]),
        mode="bicubic",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)


class WaveletColorFix:
    """
    Transfer source image/video color to target image/video while preserving target detail.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE", {
                    "tooltip": "Image or frame sequence whose detail should be preserved"
                }),
                "source_image": ("IMAGE", {
                    "tooltip": "Image or frame sequence providing the color statistics"
                }),
                "align_method": (["adain", "wavelet"], {
                    "default": "wavelet",
                    "tooltip": "AdaIN matches global color statistics; wavelet transfers low-frequency color"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "image/video"
    DESCRIPTION = "Transfer source colors to target frames using AdaIN or wavelet reconstruction"

    def process(
        self,
        target_image: torch.Tensor,
        source_image: torch.Tensor,
        align_method: str = "wavelet",
    ) -> Tuple[torch.Tensor]:
        if target_image.ndim != 4 or source_image.ndim != 4:
            raise ValueError("target_image and source_image must be IMAGE tensors in [B, H, W, C] format")

        if target_image.shape[-1] < 3 or source_image.shape[-1] < 3:
            raise ValueError("WaveletColorFix requires RGB or RGBA images")

        target_rgb = target_image[..., :3].float().clamp(0.0, 1.0)
        source_rgb = source_image[..., :3].float().clamp(0.0, 1.0)

        source_rgb = _resize_source_to_target(source_rgb, target_rgb)
        frame_count = target_rgb.shape[0]
        source_count = source_rgb.shape[0]

        if frame_count != source_count:
            logging.warning(
                "WaveletColorFix frame count mismatch: target=%s source=%s; source frames will loop",
                frame_count,
                source_count,
            )

        progress = comfy.utils.ProgressBar(frame_count)
        output_frames = []

        for index in range(frame_count):
            target_frame = target_rgb[index:index + 1].permute(0, 3, 1, 2)
            source_frame = source_rgb[index % source_count:index % source_count + 1].permute(0, 3, 1, 2)

            if align_method == "adain":
                fixed = _adaptive_instance_normalization(target_frame, source_frame)
            else:
                fixed = _wavelet_reconstruction(target_frame, source_frame)

            output_frames.append(fixed.clamp(0.0, 1.0).permute(0, 2, 3, 1))
            progress.update(1)

        output = torch.cat(output_frames, dim=0)

        if target_image.shape[-1] == 4:
            output = torch.cat([output, target_image[..., 3:4]], dim=-1)

        return (output,)
