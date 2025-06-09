"""
FrameExtender Node for ComfyUI

Extends video sequences by adding frames at configurable positions with advanced blending options.
Supports memory-efficient processing and automatic resolution matching.
"""

import torch
import logging
from typing import Tuple, Optional

try:
    from .utils import (
        validate_image_tensor,
        resize_to_match,
        advanced_resize,
        create_blend_transition,
        safe_frame_index,
        chunk_process_frames
    )
except ImportError:
    # Fallback for direct execution
    from utils import (
        validate_image_tensor,
        resize_to_match,
        advanced_resize,
        create_blend_transition,
        safe_frame_index,
        chunk_process_frames
    )

class FrameExtender:
    """
    Professional frame extension node for ComfyUI.
    
    Extends video sequences by adding frames at specified positions with smooth transitions.
    Supports various blending modes and automatic resolution matching for seamless integration.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("IMAGE", {
                    "tooltip": "Base video sequence to extend (IMAGE tensor format)"
                }),
                "additional_frames": ("IMAGE", {
                    "tooltip": "Frames to add to the source video"
                }),
                "insert_position": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Position to insert frames (-1 = end, 0 = beginning)"
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
                "blend_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Number of transition frames for smooth blending"
                }),
                "loop_additional": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loop additional frames if source is longer"
                }),
                "memory_efficient": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process in chunks to save memory"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("extended_video",)
    FUNCTION = "extend_frames"
    CATEGORY = "image/video"
    DESCRIPTION = "Extends video sequences by adding frames at configurable positions with smooth blending"
    
    def extend_frames(self, source_video: torch.Tensor, additional_frames: torch.Tensor,
                     insert_position: int = -1, width: int = 512, height: int = 512,
                     interpolation: str = "bilinear", method: str = "keep proportion",
                     condition: str = "always", multiple_of: int = 0,
                     blend_frames: int = 0, loop_additional: bool = False,
                     memory_efficient: bool = True) -> Tuple[torch.Tensor]:
        """
        Extend video sequence with additional frames.
        
        Args:
            source_video: Base video tensor [B, H, W, C]
            additional_frames: Frames to add [B, H, W, C]
            insert_position: Where to insert frames (-1 for end)
            resize_method: Interpolation method for resizing
            blend_frames: Number of transition frames
            loop_additional: Whether to loop additional frames
            memory_efficient: Process in chunks
            
        Returns:
            Extended video tensor
        """
        try:
            # Validate inputs
            validate_image_tensor(source_video, "source_video")
            validate_image_tensor(additional_frames, "additional_frames")
            
            if source_video.shape[0] == 0:
                logging.warning("Source video is empty, returning additional frames")
                return (additional_frames,)
            
            if additional_frames.shape[0] == 0:
                logging.warning("Additional frames are empty, returning source video")
                return (source_video,)
            
            # Handle insert position
            source_length = source_video.shape[0]
            if insert_position == -1:
                insert_position = source_length  # Insert at end
            else:
                # Clamp insert_position to valid range [0, source_length]
                insert_position = max(0, min(insert_position, source_length))
            
            # Apply advanced resizing to both source and additional frames
            # First, determine target dimensions
            if condition == "always" or width != source_video.shape[2] or height != source_video.shape[1]:
                # Resize source video to target dimensions
                source_video = advanced_resize(source_video, width, height, interpolation, method, condition, multiple_of)
                logging.info(f"Resized source video to {source_video.shape[1:3]}")

            # Resize additional frames to match source dimensions (after potential source resize)
            if additional_frames.shape[1:3] != source_video.shape[1:3]:
                logging.info(f"Resizing additional frames from {additional_frames.shape[1:3]} to {source_video.shape[1:3]}")
                additional_frames = advanced_resize(additional_frames, source_video.shape[2], source_video.shape[1],
                                                  interpolation, method, "always", multiple_of)
            
            # Handle channel mismatch
            if additional_frames.shape[3] != source_video.shape[3]:
                if source_video.shape[3] == 3 and additional_frames.shape[3] == 4:
                    # Remove alpha channel
                    additional_frames = additional_frames[:, :, :, :3]
                elif source_video.shape[3] == 4 and additional_frames.shape[3] == 3:
                    # Add alpha channel
                    alpha = torch.ones_like(additional_frames[:, :, :, :1])
                    additional_frames = torch.cat([additional_frames, alpha], dim=3)
                else:
                    raise ValueError(f"Incompatible channel counts: source={source_video.shape[3]}, additional={additional_frames.shape[3]}")
            
            # Handle looping if requested
            if loop_additional and source_length > additional_frames.shape[0]:
                repeat_count = (source_length // additional_frames.shape[0]) + 1
                additional_frames = additional_frames.repeat(repeat_count, 1, 1, 1)[:source_length]
            
            # Split source video at insert position
            before_frames = source_video[:insert_position] if insert_position > 0 else torch.empty(0, *source_video.shape[1:])
            after_frames = source_video[insert_position:] if insert_position < source_length else torch.empty(0, *source_video.shape[1:])

            # Simple concatenation - just stitch the sequences together
            result_parts = []

            # Add frames before insert position
            if before_frames.shape[0] > 0:
                result_parts.append(before_frames)

            # Add blend transition from before to additional (only if blend_frames > 0)
            if blend_frames > 0 and before_frames.shape[0] > 0 and additional_frames.shape[0] > 0:
                transition = create_blend_transition(before_frames, additional_frames, blend_frames)
                if transition.shape[0] > 0:
                    result_parts.append(transition)

            # Add additional frames
            if additional_frames.shape[0] > 0:
                result_parts.append(additional_frames)

            # Add blend transition from additional to after (only if blend_frames > 0)
            if blend_frames > 0 and additional_frames.shape[0] > 0 and after_frames.shape[0] > 0:
                transition = create_blend_transition(additional_frames, after_frames, blend_frames)
                if transition.shape[0] > 0:
                    result_parts.append(transition)

            # Add frames after insert position
            if after_frames.shape[0] > 0:
                result_parts.append(after_frames)
            
            # Filter out empty parts and ensure dimension consistency
            non_empty_parts = [part for part in result_parts if part.shape[0] > 0]

            if not non_empty_parts:
                # Return empty tensor with correct shape
                extended_video = torch.empty(0, *source_video.shape[1:])
            elif len(non_empty_parts) == 1:
                # Only one part, return it directly
                extended_video = non_empty_parts[0]
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
                extended_video = torch.cat(non_empty_parts, dim=0)
            
            logging.info(f"Extended video from {source_length} to {extended_video.shape[0]} frames")
            return (extended_video,)
            
        except Exception as e:
            logging.error(f"FrameExtender error: {str(e)}")
            raise RuntimeError(f"Frame extension failed: {str(e)}")
