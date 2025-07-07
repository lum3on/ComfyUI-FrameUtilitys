"""
FrameClipper Node for ComfyUI

Simple node to shorten video clips by specifying the number of frames to keep.
Clean and straightforward - just cuts the video to the specified frame count.
"""

import torch
import logging
from typing import Tuple

try:
    from .utils import validate_image_tensor
except ImportError:
    # Fallback for direct execution
    from utils import validate_image_tensor

class FrameClipper:
    """
    Simple frame clipping node for ComfyUI.
    
    Takes a video sequence and outputs only the specified number of frames.
    Perfect for shortening clips without complex processing.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("IMAGE", {
                    "tooltip": "Source video sequence to clip (IMAGE tensor format)"
                }),
                "frame_count": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Number of frames to keep from the start of the video"
                }),
            },
            "optional": {
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Starting frame index (0 = from beginning)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("clipped_video",)
    FUNCTION = "clip_frames"
    CATEGORY = "image/video"
    DESCRIPTION = "Simple node to shorten video clips by specifying frame count"
    
    def clip_frames(self, source_video: torch.Tensor, frame_count: int = 30, 
                   start_frame: int = 0) -> Tuple[torch.Tensor]:
        """
        Clip video to specified number of frames.
        
        Args:
            source_video: Source video tensor [B, H, W, C]
            frame_count: Number of frames to keep
            start_frame: Starting frame index
            
        Returns:
            Clipped video tensor
        """
        try:
            # Validate input
            validate_image_tensor(source_video, "source_video")
            
            if source_video.shape[0] == 0:
                logging.warning("Source video is empty")
                return (source_video,)
            
            source_length = source_video.shape[0]
            
            # Ensure start_frame is within bounds
            start_frame = max(0, min(start_frame, source_length - 1))
            
            # Calculate end frame
            end_frame = min(start_frame + frame_count, source_length)
            
            # Clip the video
            clipped_video = source_video[start_frame:end_frame]
            
            actual_frames = clipped_video.shape[0]
            logging.info(f"Clipped video from {source_length} frames to {actual_frames} frames "
                        f"(frames {start_frame} to {end_frame-1})")
            
            return (clipped_video,)
            
        except Exception as e:
            logging.error(f"FrameClipper error: {str(e)}")
            raise RuntimeError(f"Frame clipping failed: {str(e)}")
