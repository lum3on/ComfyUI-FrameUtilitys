"""
FrameReplacer Node for ComfyUI

Precisely replaces frames in video sequences with advanced targeting and transition options.
Supports frame-accurate replacement with smooth blending capabilities.
"""

import torch
import logging
from typing import Tuple, Optional

try:
    from .utils import (
        validate_image_tensor,
        resize_to_match,
        advanced_resize,
        blend_frames,
        safe_frame_index,
        chunk_process_frames
    )
except ImportError:
    # Fallback for direct execution
    from utils import (
        validate_image_tensor,
        resize_to_match,
        advanced_resize,
        blend_frames,
        safe_frame_index,
        chunk_process_frames
    )

class FrameReplacer:
    """
    Professional frame replacement node for ComfyUI.
    
    Replaces specific frames in video sequences with precise control and smooth transitions.
    Supports batch replacement and advanced blending for seamless integration.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("IMAGE", {
                    "tooltip": "Source video sequence to modify (IMAGE tensor format)"
                }),
                "replacement_frames": ("IMAGE", {
                    "tooltip": "Frames to use as replacements"
                }),
                "target_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Starting frame index for replacement (0-based)"
                }),
                "replace_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of frames to replace"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Target width for frame resizing"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Target height for frame resizing"
                }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {
                    "default": "bilinear",
                    "tooltip": "Interpolation method for resizing"
                }),
                "method": (["stretch", "keep proportion", "fill / crop", "pad"], {
                    "default": "keep proportion",
                    "tooltip": "Resizing method"
                }),
                "condition": (["always", "downscale if bigger", "upscale if smaller", "if different"], {
                    "default": "always",
                    "tooltip": "When to apply resizing"
                }),
                "multiple_of": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Round dimensions to multiple of this value (0 = disabled)"
                }),
            },
            "optional": {
                "blend_edges": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Smooth blend at replacement boundaries"
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Strength of edge blending (0.0 = no blend, 1.0 = full blend)"
                }),
                "loop_replacement": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Loop replacement frames if fewer than replace_count"
                }),
                "preserve_length": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Maintain original video length"
                }),
                "memory_efficient": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process in chunks to save memory"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("modified_video",)
    FUNCTION = "replace_frames"
    CATEGORY = "image/video"
    DESCRIPTION = "Precisely replaces frames in video sequences with advanced targeting and blending"
    
    def replace_frames(self, source_video: torch.Tensor, replacement_frames: torch.Tensor,
                      target_frame: int = 0, replace_count: int = 1,
                      width: int = 512, height: int = 512,
                      interpolation: str = "bilinear", method: str = "keep proportion",
                      condition: str = "always", multiple_of: int = 0,
                      blend_edges: bool = False, blend_strength: float = 0.5,
                      loop_replacement: bool = True, preserve_length: bool = True,
                      memory_efficient: bool = True) -> Tuple[torch.Tensor]:
        """
        Replace frames in video sequence with precise control.
        
        Args:
            source_video: Source video tensor [B, H, W, C]
            replacement_frames: Replacement frames [B, H, W, C]
            target_frame: Starting frame index for replacement
            replace_count: Number of frames to replace
            resize_method: Interpolation method for resizing
            blend_edges: Whether to blend at boundaries
            blend_strength: Strength of edge blending
            loop_replacement: Loop replacement frames if needed
            preserve_length: Maintain original video length
            memory_efficient: Process in chunks
            
        Returns:
            Modified video tensor
        """
        try:
            # Validate inputs
            validate_image_tensor(source_video, "source_video")
            validate_image_tensor(replacement_frames, "replacement_frames")
            
            source_length = source_video.shape[0]
            replacement_length = replacement_frames.shape[0]
            
            if source_length == 0:
                raise ValueError("Source video cannot be empty")
            
            if replacement_length == 0:
                raise ValueError("Replacement frames cannot be empty")
            
            # Validate frame indices
            target_frame = max(0, min(target_frame, source_length - 1))
            
            if target_frame + replace_count > source_length and preserve_length:
                replace_count = source_length - target_frame
                logging.warning(f"Adjusted replace_count to {replace_count} to preserve video length")
            
            if replace_count <= 0:
                logging.warning("No frames to replace, returning original video")
                return (source_video,)
            
            # Apply advanced resizing to both source and replacement frames
            # First, determine target dimensions and resize source if needed
            if condition == "always" or width != source_video.shape[2] or height != source_video.shape[1]:
                # Resize source video to target dimensions
                source_video = advanced_resize(source_video, width, height, interpolation, method, condition, multiple_of)
                logging.info(f"Resized source video to {source_video.shape[1:3]}")

            # Resize replacement frames to match source dimensions (after potential source resize)
            if replacement_frames.shape[1:3] != source_video.shape[1:3]:
                logging.info(f"Resizing replacement frames from {replacement_frames.shape[1:3]} to {source_video.shape[1:3]}")
                replacement_frames = advanced_resize(replacement_frames, source_video.shape[2], source_video.shape[1],
                                                   interpolation, method, "always", multiple_of)
            
            # Handle channel mismatch
            if replacement_frames.shape[3] != source_video.shape[3]:
                if source_video.shape[3] == 3 and replacement_frames.shape[3] == 4:
                    # Remove alpha channel
                    replacement_frames = replacement_frames[:, :, :, :3]
                elif source_video.shape[3] == 4 and replacement_frames.shape[3] == 3:
                    # Add alpha channel
                    alpha = torch.ones_like(replacement_frames[:, :, :, :1])
                    replacement_frames = torch.cat([replacement_frames, alpha], dim=3)
                else:
                    raise ValueError(f"Incompatible channel counts: source={source_video.shape[3]}, replacement={replacement_frames.shape[3]}")
            
            # Prepare replacement frames
            if loop_replacement and replacement_length < replace_count:
                # Loop replacement frames to match replace_count
                repeat_count = (replace_count // replacement_length) + 1
                replacement_frames = replacement_frames.repeat(repeat_count, 1, 1, 1)[:replace_count]
            elif replacement_length > replace_count:
                # Trim replacement frames to match replace_count
                replacement_frames = replacement_frames[:replace_count]
            elif replacement_length < replace_count and not loop_replacement:
                # Extend with last frame
                last_frame = replacement_frames[-1:].repeat(replace_count - replacement_length, 1, 1, 1)
                replacement_frames = torch.cat([replacement_frames, last_frame], dim=0)
            
            # Split source video
            before_frames = source_video[:target_frame] if target_frame > 0 else torch.empty(0, *source_video.shape[1:])
            original_replaced = source_video[target_frame:target_frame + replace_count]
            after_frames = source_video[target_frame + replace_count:] if target_frame + replace_count < source_length else torch.empty(0, *source_video.shape[1:])
            
            # Apply edge blending if requested
            final_replacement = replacement_frames
            if blend_edges and blend_strength > 0.0:
                # Blend first frame with previous frame
                if before_frames.shape[0] > 0:
                    prev_frame = before_frames[-1:]
                    first_replacement = blend_frames(prev_frame, replacement_frames[:1], blend_strength)
                    final_replacement = torch.cat([first_replacement, replacement_frames[1:]], dim=0)
                
                # Blend last frame with next frame
                if after_frames.shape[0] > 0 and final_replacement.shape[0] > 0:
                    next_frame = after_frames[:1]
                    last_replacement = blend_frames(final_replacement[-1:], next_frame, blend_strength)
                    if final_replacement.shape[0] > 1:
                        final_replacement = torch.cat([final_replacement[:-1], last_replacement], dim=0)
                    else:
                        final_replacement = last_replacement
            
            # Combine all parts
            result_parts = []
            
            if before_frames.shape[0] > 0:
                result_parts.append(before_frames)
            
            result_parts.append(final_replacement)
            
            if after_frames.shape[0] > 0:
                result_parts.append(after_frames)
            
            # Filter out empty parts and ensure dimension consistency
            non_empty_parts = [part for part in result_parts if part.shape[0] > 0]

            if not non_empty_parts:
                # Return original source video if no parts
                modified_video = source_video
            elif len(non_empty_parts) == 1:
                # Only one part, return it directly
                modified_video = non_empty_parts[0]
            else:
                # Multiple parts - ensure they all have the same dimensions
                target_shape = source_video.shape[1:]  # Use source video dimensions as reference

                # Resize all parts to match target dimensions if needed
                for i, part in enumerate(non_empty_parts):
                    if part.shape[1:] != target_shape:
                        logging.warning(f"Part {i} has shape {part.shape[1:]}, resizing to {target_shape}")
                        # Use advanced_resize to ensure proper dimensions
                        non_empty_parts[i] = advanced_resize(part, target_shape[1], target_shape[0],
                                                           interpolation, "stretch", "always", 0)

                # Now concatenate with matching dimensions
                modified_video = torch.cat(non_empty_parts, dim=0)
            
            logging.info(f"Replaced {replace_count} frames starting at frame {target_frame}")
            return (modified_video,)
            
        except Exception as e:
            logging.error(f"FrameReplacer error: {str(e)}")
            raise RuntimeError(f"Frame replacement failed: {str(e)}")
