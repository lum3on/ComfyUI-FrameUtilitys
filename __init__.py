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
    from .frame_repeater import FrameRepeater
    from .frame_clipper import FrameClipper
    from .git_installer import GitInstaller
    from .wavelet_color_fix import WaveletColorFix
    from .multiply_sigmas import MultiplySigmas
except ImportError:
    from frame_extender import FrameExtender
    from frame_extender_advanced import FrameExtenderAdvanced
    from frame_replacer import FrameReplacer
    from frame_repeater import FrameRepeater
    from frame_clipper import FrameClipper
    from git_installer import GitInstaller
    from wavelet_color_fix import WaveletColorFix
    from multiply_sigmas import MultiplySigmas

# Node class mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "FrameExtender": FrameExtender,
    "FrameExtenderAdvanced": FrameExtenderAdvanced,
    "FrameReplacer": FrameReplacer,
    "FrameRepeater": FrameRepeater,
    "FrameClipper": FrameClipper,
    "GitInstaller": GitInstaller,
    "WaveletColorFix": WaveletColorFix,
    "MultiplySigmas": MultiplySigmas,
}

# Display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameExtender": "Frame Extender 🎬",
    "FrameExtenderAdvanced": "Frame Extender Advanced 🎭",
    "FrameReplacer": "Frame Replacer ✂️",
    "FrameRepeater": "Frame Repeater 🔁",
    "FrameClipper": "Frame Clipper 📹",
    "GitInstaller": "Git Repository Installer 📦",
    "WaveletColorFix": "Wavelet Color Fix",
    "MultiplySigmas": "Multiply Sigmas (stateless)",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Version info
__version__ = "1.1.0"
__author__ = "ComfyUI Node Architect"
__description__ = "Professional frame manipulation tools and development utilities for ComfyUI"
