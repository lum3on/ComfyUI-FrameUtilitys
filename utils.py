"""
Utility functions for Frame Utility nodes.

Provides shared functionality for frame manipulation, validation, and processing.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import logging

def validate_image_tensor(tensor: torch.Tensor, name: str) -> None:
    """
    Validate that a tensor is a proper IMAGE tensor for ComfyUI.
    
    Args:
        tensor: Input tensor to validate
        name: Name of the tensor for error reporting
        
    Raises:
        ValueError: If tensor format is invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if len(tensor.shape) != 4:
        raise ValueError(f"{name} must be 4D tensor [B, H, W, C], got shape {tensor.shape}")
    
    if tensor.shape[-1] not in [1, 3, 4]:  # Grayscale, RGB, or RGBA
        raise ValueError(f"{name} must have 1, 3, or 4 channels, got {tensor.shape[-1]}")
    
    if tensor.dtype not in [torch.float32, torch.float16]:
        logging.warning(f"{name} has dtype {tensor.dtype}, converting to float32")

def resize_to_match(source: torch.Tensor, target: torch.Tensor,
                   method: str = "bilinear") -> torch.Tensor:
    """
    Resize source tensor to match target tensor dimensions.

    Args:
        source: Source tensor to resize [B, H, W, C]
        target: Target tensor to match dimensions [B, H, W, C]
        method: Interpolation method ('nearest', 'bilinear', 'bicubic', 'area')

    Returns:
        Resized source tensor matching target dimensions
    """
    if source.shape[1:3] == target.shape[1:3]:
        return source

    # Convert to [B, C, H, W] for F.interpolate
    source_reshaped = source.permute(0, 3, 1, 2)

    # Resize to target dimensions
    resized = F.interpolate(
        source_reshaped,
        size=(target.shape[1], target.shape[2]),
        mode=method,
        align_corners=False if method in ['bilinear', 'bicubic'] else None
    )

    # Convert back to [B, H, W, C]
    return resized.permute(0, 2, 3, 1)

def advanced_resize(frames: torch.Tensor, width: int, height: int,
                   interpolation: str = "bilinear", method: str = "keep proportion",
                   condition: str = "always", multiple_of: int = 0) -> torch.Tensor:
    """
    Advanced frame resizing with multiple methods and conditions.

    Args:
        frames: Input frames [B, H, W, C]
        width: Target width
        height: Target height
        interpolation: Interpolation method
        method: Resizing method ('stretch', 'keep proportion', 'fill / crop', 'pad')
        condition: When to resize ('always', 'downscale if bigger', 'upscale if smaller', 'if different')
        multiple_of: Round dimensions to multiple of this value

    Returns:
        Resized frames
    """
    current_height, current_width = frames.shape[1], frames.shape[2]

    # Apply multiple_of constraint
    if multiple_of > 0:
        width = ((width + multiple_of - 1) // multiple_of) * multiple_of
        height = ((height + multiple_of - 1) // multiple_of) * multiple_of

    # Check condition
    should_resize = True
    if condition == "downscale if bigger":
        should_resize = current_width > width or current_height > height
    elif condition == "upscale if smaller":
        should_resize = current_width < width or current_height < height
    elif condition == "if different":
        should_resize = current_width != width or current_height != height

    if not should_resize:
        return frames

    # Calculate target dimensions based on method
    if method == "stretch":
        target_width, target_height = width, height
    elif method == "keep proportion":
        # Maintain aspect ratio, fit within target dimensions
        scale = min(width / current_width, height / current_height)
        target_width = int(current_width * scale)
        target_height = int(current_height * scale)

        # Apply multiple_of constraint to final dimensions
        if multiple_of > 0:
            target_width = ((target_width + multiple_of - 1) // multiple_of) * multiple_of
            target_height = ((target_height + multiple_of - 1) // multiple_of) * multiple_of

    elif method == "fill / crop":
        # Fill target dimensions, crop if necessary
        scale = max(width / current_width, height / current_height)
        target_width = int(current_width * scale)
        target_height = int(current_height * scale)
    elif method == "pad":
        # Keep original size, will pad later
        target_width, target_height = current_width, current_height
    else:
        target_width, target_height = width, height

    # Perform resize
    frames_reshaped = frames.permute(0, 3, 1, 2)  # [B, C, H, W]

    # Handle different interpolation methods
    mode = interpolation
    if interpolation == "lanczos":
        # PyTorch doesn't have lanczos, use bicubic as fallback
        mode = "bicubic"

    resized = F.interpolate(
        frames_reshaped,
        size=(target_height, target_width),
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )

    resized = resized.permute(0, 2, 3, 1)  # [B, H, W, C]

    # Handle padding or cropping for specific methods
    if method == "pad" and (target_width != width or target_height != height):
        # Pad to target dimensions
        pad_h = max(0, height - target_height)
        pad_w = max(0, width - target_width)

        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # Pad with zeros (black)
            resized = F.pad(resized.permute(0, 3, 1, 2),
                          (pad_left, pad_right, pad_top, pad_bottom),
                          mode='constant', value=0.0).permute(0, 2, 3, 1)

    elif method == "fill / crop" and (target_width != width or target_height != height):
        # Crop to target dimensions
        if target_height > height or target_width > width:
            crop_h = min(target_height, height)
            crop_w = min(target_width, width)

            start_h = max(0, (target_height - crop_h) // 2)
            start_w = max(0, (target_width - crop_w) // 2)

            end_h = min(target_height, start_h + crop_h)
            end_w = min(target_width, start_w + crop_w)

            resized = resized[:, start_h:end_h, start_w:end_w, :]

    return resized

def blend_frames(frame1: torch.Tensor, frame2: torch.Tensor, 
                alpha: float) -> torch.Tensor:
    """
    Blend two frames with alpha blending.
    
    Args:
        frame1: First frame [1, H, W, C]
        frame2: Second frame [1, H, W, C]
        alpha: Blending factor (0.0 = frame1, 1.0 = frame2)
        
    Returns:
        Blended frame
    """
    alpha = torch.clamp(torch.tensor(alpha), 0.0, 1.0)
    return frame1 * (1.0 - alpha) + frame2 * alpha

def create_blend_transition(frames1: torch.Tensor, frames2: torch.Tensor,
                          blend_length: int) -> torch.Tensor:
    """
    Create smooth transition between two frame sequences.

    Args:
        frames1: First sequence [B1, H, W, C]
        frames2: Second sequence [B2, H, W, C]
        blend_length: Number of frames to blend

    Returns:
        Transition frames [blend_length, H, W, C]
    """
    if blend_length <= 0:
        return torch.empty(0, frames1.shape[1], frames1.shape[2], frames1.shape[3])

    # Get last frame of first sequence and first frame of second sequence
    last_frame = frames1[-1:] if frames1.shape[0] > 0 else frames2[:1]
    first_frame = frames2[:1] if frames2.shape[0] > 0 else frames1[-1:]

    # Ensure frames match dimensions exactly
    if last_frame.shape != first_frame.shape:
        # Resize first_frame to match last_frame exactly
        first_frame = resize_to_match(first_frame, last_frame)

        # Double check dimensions match
        if last_frame.shape[1:] != first_frame.shape[1:]:
            # Force exact match by creating new tensor with correct dimensions
            target_shape = last_frame.shape
            new_first_frame = torch.zeros_like(last_frame)

            # Copy data with safe bounds
            min_h = min(first_frame.shape[1], target_shape[1])
            min_w = min(first_frame.shape[2], target_shape[2])
            min_c = min(first_frame.shape[3], target_shape[3])

            new_first_frame[:, :min_h, :min_w, :min_c] = first_frame[:, :min_h, :min_w, :min_c]
            first_frame = new_first_frame

    # Create blend sequence
    blend_frames_list = []
    for i in range(blend_length):
        alpha = (i + 1) / (blend_length + 1)
        blended = blend_frames(last_frame, first_frame, alpha)
        blend_frames_list.append(blended)

    return torch.cat(blend_frames_list, dim=0) if blend_frames_list else torch.empty(0, last_frame.shape[1], last_frame.shape[2], last_frame.shape[3])

def safe_frame_index(frame_count: int, index: int, allow_negative: bool = True) -> int:
    """
    Convert frame index to safe positive index with bounds checking.
    
    Args:
        frame_count: Total number of frames
        index: Frame index (can be negative)
        allow_negative: Whether to allow negative indexing
        
    Returns:
        Safe positive index
        
    Raises:
        ValueError: If index is out of bounds
    """
    if frame_count <= 0:
        raise ValueError("Frame count must be positive")
    
    if allow_negative and index < 0:
        index = frame_count + index
    
    if index < 0 or index >= frame_count:
        raise ValueError(f"Frame index {index} out of bounds for {frame_count} frames")
    
    return index

def chunk_process_frames(frames: torch.Tensor, chunk_size: int = 32) -> list:
    """
    Split frames into chunks for memory-efficient processing.

    Args:
        frames: Input frames [B, H, W, C]
        chunk_size: Maximum frames per chunk

    Returns:
        List of frame chunks
    """
    if chunk_size <= 0:
        chunk_size = frames.shape[0]

    chunks = []
    for i in range(0, frames.shape[0], chunk_size):
        chunk = frames[i:i + chunk_size]
        chunks.append(chunk)

    return chunks




