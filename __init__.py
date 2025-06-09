"""
ComfyUI Frame Utility Nodes

A comprehensive collection of nodes for advanced frame manipulation in ComfyUI.
Provides professional-grade video editing capabilities with native IMAGE tensor support.

Author: ComfyUI Node Architect
Version: 1.0.0
License: MIT
"""

try:
    from .frame_extender import FrameExtender
    from .frame_extender_advanced import FrameExtenderAdvanced
    from .frame_replacer import FrameReplacer
except ImportError:
    from frame_extender import FrameExtender
    from frame_extender_advanced import FrameExtenderAdvanced
    from frame_replacer import FrameReplacer

# Node class mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "FrameExtender": FrameExtender,
    "FrameExtenderAdvanced": FrameExtenderAdvanced,
    "FrameReplacer": FrameReplacer,
}

# Display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameExtender": "Frame Extender 🎬",
    "FrameExtenderAdvanced": "Frame Extender Advanced 🎭",
    "FrameReplacer": "Frame Replacer ✂️",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Version info
__version__ = "1.0.0"
__author__ = "ComfyUI Node Architect"
__description__ = "Professional frame manipulation tools for ComfyUI"
