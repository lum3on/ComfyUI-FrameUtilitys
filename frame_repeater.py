"""
FrameRepeater Node for ComfyUI

Repeats a specified batch of frames from a video sequence at configurable positions.
Supports flexible frame range selection and repetition patterns for creative video effects.
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

class FrameRepeater:
    """
    Professional frame repetition node for ComfyUI.
    
    Repeats a specified batch of frames from a video sequence with configurable
    positioning and repetition patterns. Perfect for creating loops, emphasis effects,
    or extending specific moments in video content.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("IMAGE", {
                    "tooltip": "Source video sequence to process (IMAGE tensor format)"
                }),
                "start_frame": ("INT", {
                    "default": -15,
                    "min": -10000,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Starting frame for repetition (negative values count from end)"
                }),
                "frame_count": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of frames to repeat"
                }),
                "repeat_times": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "How many times to repeat the frame batch"
                }),
                "output_mode": (["extract_only", "insert_into_video"], {
                    "default": "extract_only",
                    "tooltip": "Extract only the repeated frames or insert them into the full video"
                }),
                "insert_position": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Where to insert repeated frames (-1 = end, 0 = beginning) [only for insert_into_video mode]"
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
                "reverse_repeat": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse the repeated frames for ping-pong effect"
                }),
                "memory_efficient": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process in chunks to save memory"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("repeated_video",)
    FUNCTION = "repeat_frames"
    CATEGORY = "image/video"
    DESCRIPTION = "Repeats a specified batch of frames from a video sequence with configurable positioning"
    
    def repeat_frames(self, source_video: torch.Tensor, start_frame: int = -15,
                     frame_count: int = 15, repeat_times: int = 2, output_mode: str = "extract_only",
                     insert_position: int = -1, width: int = 512, height: int = 512,
                     interpolation: str = "bilinear", method: str = "keep proportion",
                     condition: str = "always", multiple_of: int = 0, blend_frames: int = 0,
                     reverse_repeat: bool = False, memory_efficient: bool = True) -> Tuple[torch.Tensor]:
        """
        Repeat a batch of frames from the video sequence.

        Args:
            source_video: Source video tensor [B, H, W, C]
            start_frame: Starting frame index for repetition (negative = from end)
            frame_count: Number of frames to repeat
            repeat_times: How many times to repeat the batch
            output_mode: "extract_only" returns just repeated frames, "insert_into_video" returns full video
            insert_position: Where to insert repeated frames (-1 for end)
            width, height: Target dimensions
            interpolation: Interpolation method
            method: Resizing method
            condition: When to resize
            multiple_of: Dimension constraint
            blend_frames: Number of transition frames
            reverse_repeat: Reverse frames for ping-pong effect
            memory_efficient: Process in chunks

        Returns:
            Video tensor with repeated frames (extract_only) or full video with insertions
        """
        try:
            # Validate inputs
            validate_image_tensor(source_video, "source_video")
            
            if source_video.shape[0] == 0:
                logging.warning("Source video is empty")
                return (source_video,)
            
            source_length = source_video.shape[0]
            
            # Convert start_frame to positive index
            if start_frame < 0:
                start_frame = max(0, source_length + start_frame)
            else:
                start_frame = min(start_frame, source_length - 1)
            
            # Ensure we don't exceed video bounds
            end_frame = min(start_frame + frame_count, source_length)
            actual_frame_count = end_frame - start_frame
            
            if actual_frame_count <= 0:
                logging.warning("No frames to repeat, returning original video")
                return (source_video,)
            
            # Apply resizing if needed
            if condition == "always" or width != source_video.shape[2] or height != source_video.shape[1]:
                source_video = advanced_resize(source_video, width, height, interpolation, method, condition, multiple_of)
                logging.info(f"Resized source video to {source_video.shape[1:3]}")
            
            # Extract frames to repeat
            frames_to_repeat = source_video[start_frame:end_frame]
            logging.info(f"Extracting frames {start_frame} to {end_frame-1} for repetition")
            
            # Create repeated sequence
            repeated_sequences = []
            for i in range(repeat_times):
                if reverse_repeat and i % 2 == 1:
                    # Reverse every other repetition for ping-pong effect
                    repeated_sequences.append(torch.flip(frames_to_repeat, dims=[0]))
                else:
                    repeated_sequences.append(frames_to_repeat)
            
            # Concatenate all repetitions
            all_repeated_frames = torch.cat(repeated_sequences, dim=0)

            # Handle different output modes
            if output_mode == "extract_only":
                # Return only the repeated frames
                logging.info(f"Extracted and repeated {actual_frame_count} frames {repeat_times} times, "
                            f"output contains {all_repeated_frames.shape[0]} frames")
                return (all_repeated_frames,)

            else:  # insert_into_video mode
                # Handle insert position
                if insert_position == -1:
                    insert_position = source_length  # Insert at end
                else:
                    insert_position = max(0, min(insert_position, source_length))

                # Split source video at insert position
                before_frames = source_video[:insert_position] if insert_position > 0 else torch.empty(0, *source_video.shape[1:])
                after_frames = source_video[insert_position:] if insert_position < source_length else torch.empty(0, *source_video.shape[1:])

                # Build result with optional blending
                result_parts = []

                # Add frames before insert position
                if before_frames.shape[0] > 0:
                    result_parts.append(before_frames)

                # Add blend transition from before to repeated frames
                if blend_frames > 0 and before_frames.shape[0] > 0 and all_repeated_frames.shape[0] > 0:
                    transition = create_blend_transition(before_frames, all_repeated_frames, blend_frames)
                    if transition.shape[0] > 0:
                        result_parts.append(transition)

                # Add repeated frames
                if all_repeated_frames.shape[0] > 0:
                    result_parts.append(all_repeated_frames)

                # Add blend transition from repeated frames to after
                if blend_frames > 0 and all_repeated_frames.shape[0] > 0 and after_frames.shape[0] > 0:
                    transition = create_blend_transition(all_repeated_frames, after_frames, blend_frames)
                    if transition.shape[0] > 0:
                        result_parts.append(transition)

                # Add frames after insert position
                if after_frames.shape[0] > 0:
                    result_parts.append(after_frames)

                # Filter out empty parts and concatenate
                non_empty_parts = [part for part in result_parts if part.shape[0] > 0]

                if not non_empty_parts:
                    # Return empty tensor with correct shape
                    repeated_video = torch.empty(0, *source_video.shape[1:])
                elif len(non_empty_parts) == 1:
                    repeated_video = non_empty_parts[0]
                else:
                    # Ensure all parts have matching dimensions
                    target_shape = source_video.shape[1:]
                    for i, part in enumerate(non_empty_parts):
                        if part.shape[1:] != target_shape:
                            logging.warning(f"Part {i} has shape {part.shape[1:]}, resizing to {target_shape}")
                            non_empty_parts[i] = advanced_resize(part, target_shape[1], target_shape[0],
                                                               interpolation, "stretch", "always", 0)

                    repeated_video = torch.cat(non_empty_parts, dim=0)

                logging.info(f"Repeated {actual_frame_count} frames {repeat_times} times, "
                            f"extended video from {source_length} to {repeated_video.shape[0]} frames")

                return (repeated_video,)
            
        except Exception as e:
            logging.error(f"FrameRepeater error: {str(e)}")
            raise RuntimeError(f"Frame repetition failed: {str(e)}")
