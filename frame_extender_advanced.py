"""
FrameExtenderAdvanced Node for ComfyUI

Advanced frame extension with comprehensive blending modes and transition effects.
Provides professional-grade video editing capabilities with multiple blend modes.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Tuple

try:
    from .utils import (
        validate_image_tensor, 
        advanced_resize,
        safe_frame_index,
        chunk_process_frames
    )
except ImportError:
    # Fallback for direct execution
    from utils import (
        validate_image_tensor, 
        advanced_resize,
        safe_frame_index,
        chunk_process_frames
    )

class FrameExtenderAdvanced:
    """
    Advanced frame extension node with comprehensive blending modes.
    
    Provides professional video editing capabilities with multiple transition effects,
    blend modes, and advanced control over frame stitching operations.
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
                "blend_mode": (["none", "linear", "ease_in", "ease_out", "ease_in_out", "crossfade", "dissolve", "overlay"], {
                    "default": "none",
                    "tooltip": "Blending mode for transitions"
                }),
                "blend_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of transition frames for blending"
                }),
                "blend_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Strength of blending effect"
                }),
                "transition_curve": (["linear", "smooth", "sharp", "bounce", "elastic"], {
                    "default": "smooth",
                    "tooltip": "Transition curve for blending"
                }),
            },
            "optional": {
                "loop_additional": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loop additional frames if source is longer"
                }),
                "reverse_additional": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse additional frames before adding"
                }),
                "fade_edges": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply fade effect at sequence edges"
                }),
                "memory_efficient": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process in chunks to save memory"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("extended_video",)
    FUNCTION = "extend_frames_advanced"
    CATEGORY = "image/video"
    DESCRIPTION = "Advanced frame extension with comprehensive blending modes and transition effects"
    
    def apply_transition_curve(self, t: float, curve_type: str) -> float:
        """Apply transition curve to interpolation factor."""
        if curve_type == "linear":
            return t
        elif curve_type == "smooth":
            return t * t * (3.0 - 2.0 * t)  # Smoothstep
        elif curve_type == "sharp":
            return t * t * t  # Cubic
        elif curve_type == "bounce":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) * (1 - t)
        elif curve_type == "elastic":
            if t == 0 or t == 1:
                return t
            import math
            return math.sin(t * 3.14159 * 2) * 0.1 + t
        return t
    
    def create_advanced_blend(self, frame1: torch.Tensor, frame2: torch.Tensor, 
                            alpha: float, blend_mode: str, strength: float) -> torch.Tensor:
        """Create advanced blend between two frames."""
        alpha = torch.clamp(torch.tensor(alpha * strength), 0.0, 1.0)
        
        if blend_mode == "none" or blend_mode == "linear":
            return frame1 * (1.0 - alpha) + frame2 * alpha
        elif blend_mode == "crossfade":
            # Smooth crossfade with gamma correction
            gamma = 2.2
            frame1_gamma = torch.pow(frame1, gamma)
            frame2_gamma = torch.pow(frame2, gamma)
            blended = frame1_gamma * (1.0 - alpha) + frame2_gamma * alpha
            return torch.pow(blended, 1.0 / gamma)
        elif blend_mode == "dissolve":
            # Random dissolve effect
            noise = torch.rand_like(frame1)
            mask = (noise < alpha).float()
            return frame1 * (1.0 - mask) + frame2 * mask
        elif blend_mode == "overlay":
            # Overlay blend mode
            mask = frame1 < 0.5
            result = torch.where(mask, 
                               2 * frame1 * frame2,
                               1 - 2 * (1 - frame1) * (1 - frame2))
            return frame1 * (1.0 - alpha) + result * alpha
        else:
            return frame1 * (1.0 - alpha) + frame2 * alpha

    def create_advanced_transition(self, frames1: torch.Tensor, frames2: torch.Tensor,
                                 blend_frames: int, blend_mode: str, blend_strength: float,
                                 transition_curve: str) -> torch.Tensor:
        """Create advanced transition between two frame sequences."""
        if blend_frames <= 0:
            return torch.empty(0, frames1.shape[1], frames1.shape[2], frames1.shape[3])

        # Get transition frames
        last_frame = frames1[-1:] if frames1.shape[0] > 0 else frames2[:1]
        first_frame = frames2[:1] if frames2.shape[0] > 0 else frames1[-1:]

        # Ensure frames match dimensions exactly
        if last_frame.shape != first_frame.shape:
            # Resize to match
            target_h, target_w = last_frame.shape[1], last_frame.shape[2]
            first_frame = advanced_resize(first_frame, target_w, target_h, "bilinear", "stretch", "always", 0)

        # Create transition sequence
        transition_frames = []
        for i in range(blend_frames):
            # Calculate interpolation factor with curve
            t = (i + 1) / (blend_frames + 1)
            curved_t = self.apply_transition_curve(t, transition_curve)

            # Apply advanced blending
            blended = self.create_advanced_blend(last_frame, first_frame, curved_t, blend_mode, blend_strength)
            transition_frames.append(blended)

        return torch.cat(transition_frames, dim=0) if transition_frames else torch.empty(0, last_frame.shape[1], last_frame.shape[2], last_frame.shape[3])

    def apply_fade_effect(self, frames: torch.Tensor, fade_type: str = "in") -> torch.Tensor:
        """Apply fade in/out effect to frame sequence."""
        if frames.shape[0] == 0:
            return frames

        fade_length = min(5, frames.shape[0] // 2)  # Fade 5 frames or half the sequence
        result = frames.clone()

        for i in range(fade_length):
            if fade_type == "in":
                alpha = i / fade_length
                result[i] = result[i] * alpha
            elif fade_type == "out":
                alpha = (fade_length - i) / fade_length
                result[-(i+1)] = result[-(i+1)] * alpha

        return result

    def extend_frames_advanced(self, source_video: torch.Tensor, additional_frames: torch.Tensor,
                             insert_position: int = -1, width: int = 512, height: int = 512,
                             interpolation: str = "bilinear", method: str = "keep proportion",
                             condition: str = "always", multiple_of: int = 0,
                             blend_mode: str = "none", blend_frames: int = 0,
                             blend_strength: float = 1.0, transition_curve: str = "smooth",
                             loop_additional: bool = False, reverse_additional: bool = False,
                             fade_edges: bool = False, memory_efficient: bool = True) -> Tuple[torch.Tensor]:
        """
        Advanced frame extension with comprehensive blending and effects.

        Args:
            source_video: Base video tensor [B, H, W, C]
            additional_frames: Frames to add [B, H, W, C]
            insert_position: Where to insert frames (-1 for end)
            width, height: Target dimensions
            interpolation: Resizing interpolation method
            method: Resizing method
            condition: When to resize
            multiple_of: Dimension constraint
            blend_mode: Blending mode for transitions
            blend_frames: Number of transition frames
            blend_strength: Strength of blending effect
            transition_curve: Transition curve type
            loop_additional: Loop additional frames
            reverse_additional: Reverse additional frames
            fade_edges: Apply fade effects
            memory_efficient: Process in chunks

        Returns:
            Extended video tensor with advanced effects
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

            # Apply advanced resizing to both sequences
            if condition == "always" or width != source_video.shape[2] or height != source_video.shape[1]:
                source_video = advanced_resize(source_video, width, height, interpolation, method, condition, multiple_of)
                logging.info(f"Resized source video to {source_video.shape[1:3]}")

            if additional_frames.shape[1:3] != source_video.shape[1:3]:
                logging.info(f"Resizing additional frames from {additional_frames.shape[1:3]} to {source_video.shape[1:3]}")
                additional_frames = advanced_resize(additional_frames, source_video.shape[2], source_video.shape[1],
                                                  interpolation, method, "always", multiple_of)

            # Handle channel mismatch
            if additional_frames.shape[3] != source_video.shape[3]:
                if source_video.shape[3] == 3 and additional_frames.shape[3] == 4:
                    additional_frames = additional_frames[:, :, :, :3]
                elif source_video.shape[3] == 4 and additional_frames.shape[3] == 3:
                    alpha = torch.ones_like(additional_frames[:, :, :, :1])
                    additional_frames = torch.cat([additional_frames, alpha], dim=3)
                else:
                    raise ValueError(f"Incompatible channel counts: source={source_video.shape[3]}, additional={additional_frames.shape[3]}")

            # Apply additional frame modifications
            if reverse_additional:
                additional_frames = torch.flip(additional_frames, dims=[0])
                logging.info("Reversed additional frames")

            if loop_additional and source_length > additional_frames.shape[0]:
                repeat_count = (source_length // additional_frames.shape[0]) + 1
                additional_frames = additional_frames.repeat(repeat_count, 1, 1, 1)[:source_length]
                logging.info(f"Looped additional frames to {additional_frames.shape[0]} frames")

            # Apply fade effects if requested
            if fade_edges:
                additional_frames = self.apply_fade_effect(additional_frames, "in")
                logging.info("Applied fade-in effect to additional frames")

            # Split source video at insert position
            before_frames = source_video[:insert_position] if insert_position > 0 else torch.empty(0, *source_video.shape[1:])
            after_frames = source_video[insert_position:] if insert_position < source_length else torch.empty(0, *source_video.shape[1:])

            # Build result with advanced transitions
            result_parts = []

            # Add frames before insert position
            if before_frames.shape[0] > 0:
                result_parts.append(before_frames)

            # Add advanced transition from before to additional
            if blend_frames > 0 and before_frames.shape[0] > 0 and additional_frames.shape[0] > 0:
                transition = self.create_advanced_transition(before_frames, additional_frames,
                                                           blend_frames, blend_mode, blend_strength, transition_curve)
                if transition.shape[0] > 0:
                    result_parts.append(transition)

            # Add additional frames
            if additional_frames.shape[0] > 0:
                result_parts.append(additional_frames)

            # Add advanced transition from additional to after
            if blend_frames > 0 and additional_frames.shape[0] > 0 and after_frames.shape[0] > 0:
                transition = self.create_advanced_transition(additional_frames, after_frames,
                                                           blend_frames, blend_mode, blend_strength, transition_curve)
                if transition.shape[0] > 0:
                    result_parts.append(transition)

            # Add frames after insert position
            if after_frames.shape[0] > 0:
                result_parts.append(after_frames)

            # Filter out empty parts and ensure dimension consistency
            non_empty_parts = [part for part in result_parts if part.shape[0] > 0]

            if not non_empty_parts:
                extended_video = torch.empty(0, *source_video.shape[1:])
            elif len(non_empty_parts) == 1:
                extended_video = non_empty_parts[0]
            else:
                # Ensure all parts have matching dimensions
                target_shape = source_video.shape[1:]
                for i, part in enumerate(non_empty_parts):
                    if part.shape[1:] != target_shape:
                        logging.warning(f"Part {i} has shape {part.shape[1:]}, resizing to {target_shape}")
                        non_empty_parts[i] = advanced_resize(part, target_shape[1], target_shape[0],
                                                           interpolation, "stretch", "always", 0)

                extended_video = torch.cat(non_empty_parts, dim=0)

            logging.info(f"Advanced extension: {source_length} → {extended_video.shape[0]} frames with {blend_mode} blending")
            return (extended_video,)

        except Exception as e:
            logging.error(f"FrameExtenderAdvanced error: {str(e)}")
            raise RuntimeError(f"Advanced frame extension failed: {str(e)}")
