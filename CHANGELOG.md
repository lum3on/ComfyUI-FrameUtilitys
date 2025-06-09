# Changelog

All notable changes to ComfyUI Frame Utility will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-XX

### Added
- **FrameExtender**: Professional frame extension with flexible positioning
  - Insert frames at any position (beginning, end, or middle)
  - Smart blending with smooth transitions
  - Advanced resizing with multiple methods and conditions
  - Memory-efficient chunk processing
  - Loop support for additional frames

- **FrameExtenderAdvanced**: Enhanced version with professional effects
  - Multiple blend modes: linear, crossfade, dissolve, overlay
  - Transition curves: linear, smooth, sharp, bounce, elastic
  - Advanced effects: fade edges, reverse sequences
  - Comprehensive blend strength control

- **FrameReplacer**: Precise frame replacement tool
  - Frame-accurate replacement with index control
  - Batch replacement for multiple consecutive frames
  - Edge blending for smooth transitions
  - Flexible modes: loop or extend replacement frames
  - Length preservation option

- **Advanced Resizing System**
  - Multiple methods: stretch, keep proportion, fill/crop, pad
  - Smart conditions: always, downscale if bigger, upscale if smaller, if different
  - Interpolation options: nearest, bilinear, bicubic, area, lanczos
  - Dimension constraints with multiple support

- **Utility Functions**
  - Comprehensive tensor validation
  - Advanced resizing algorithms
  - Smooth blending and transition creation
  - Memory-efficient chunk processing
  - Safe frame indexing with bounds checking

### Technical Features
- Native ComfyUI IMAGE tensor support
- Production-grade error handling
- Memory optimization for large video sequences
- Comprehensive input validation
- Graceful degradation for edge cases

### Documentation
- Comprehensive README with usage examples
- Advanced usage guide
- Workflow examples
- Technical documentation
- MIT License
