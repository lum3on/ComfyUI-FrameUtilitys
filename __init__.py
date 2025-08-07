"""
ComfyUI Frame Utility Nodes

A comprehensive collection of nodes for advanced frame manipulation and development tools in ComfyUI.
Provides professional-grade video editing capabilities with native IMAGE tensor support,
plus automated GitHub repository installation for seamless custom node management.

Author: ComfyUI Node Architect
Version: 1.1.0
License: MIT
"""

try:
    from .frame_extender import FrameExtender
    from .frame_extender_advanced import FrameExtenderAdvanced
    from .frame_replacer import FrameReplacer
    from .git_installer import GitInstaller
except ImportError:
    from frame_extender import FrameExtender
    from frame_extender_advanced import FrameExtenderAdvanced
    from frame_replacer import FrameReplacer
    from git_installer import GitInstaller

# Node class mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "FrameExtender": FrameExtender,
    "FrameExtenderAdvanced": FrameExtenderAdvanced,
    "FrameReplacer": FrameReplacer,
    "GitInstaller": GitInstaller,
}

# Display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameExtender": "Frame Extender 🎬",
    "FrameExtenderAdvanced": "Frame Extender Advanced 🎭",
    "FrameReplacer": "Frame Replacer ✂️",
    "GitInstaller": "Git Repository Installer 📦",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Version info
__version__ = "1.1.0"
__author__ = "ComfyUI Node Architect"
__description__ = "Professional frame manipulation tools and development utilities for ComfyUI"
