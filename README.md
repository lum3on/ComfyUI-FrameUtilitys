# ComfyUI Frame Utility Nodes 🎬

Professional-grade frame manipulation tools for ComfyUI, providing advanced video editing capabilities with native IMAGE tensor support.

## 🌟 Features

### 🎯 **Frame Extender**
- **Flexible Positioning**: Insert frames at any position (beginning, end, or middle)
- **Smart Blending**: Smooth transitions between frame sequences
- **Advanced Resizing**: Professional-grade resizing with multiple methods and conditions
- **Memory Efficient**: Chunk processing for large video sequences
- **Loop Support**: Loop additional frames to match source length

### 🎭 **Frame Extender Advanced**
- **Professional Blend Modes**: linear, crossfade, dissolve, overlay effects
- **Transition Curves**: linear, smooth, sharp, bounce, elastic timing
- **Advanced Effects**: fade edges, reverse sequences, blend strength control
- **Creative Control**: Comprehensive parameters for artistic video editing
- **All Standard Features**: Includes all Frame Extender capabilities plus advanced effects

### ✂️ **Frame Replacer**
- **Precise Targeting**: Frame-accurate replacement with index control
- **Batch Replacement**: Replace multiple consecutive frames
- **Edge Blending**: Smooth transitions at replacement boundaries
- **Advanced Resizing**: Professional-grade resizing with multiple methods and conditions
- **Flexible Modes**: Loop or extend replacement frames as needed
- **Length Preservation**: Maintain original video length option

### 🔧 **Advanced Resizing Features**
- **Multiple Methods**: stretch, keep proportion, fill/crop, pad
- **Smart Conditions**: always, downscale if bigger, upscale if smaller, if different
- **Interpolation Options**: nearest, bilinear, bicubic, area, lanczos
- **Dimension Constraints**: Round to multiples for compatibility
- **Precision Control**: 8-pixel step increments for optimal results

## Installation

### Method 1: Git Clone (Recommended)
1. Navigate to your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/yourusername/ComfyUI-FrameUtillity.git
```

3. Restart ComfyUI - the nodes will be automatically registered.

### Method 2: Manual Download
1. Download the repository as a ZIP file
2. Extract to `ComfyUI/custom_nodes/ComfyUI-FrameUtillity/`
3. Restart ComfyUI

### Requirements
- ComfyUI (latest version recommended)
- PyTorch (included with ComfyUI)
- No additional dependencies required!

## Node Reference

### Frame Extender 🎬

**Category**: `image/video`

**Inputs**:
- `source_video` (IMAGE): Base video sequence to extend
- `additional_frames` (IMAGE): Frames to add to the source
- `insert_position` (INT): Position to insert frames (-1 = end, 0 = beginning)
- `width` (INT): Target width for frame resizing (64-8192, step 8)
- `height` (INT): Target height for frame resizing (64-8192, step 8)
- `interpolation` (COMBO): Interpolation method (nearest, bilinear, bicubic, area, lanczos)
- `method` (COMBO): Resizing method (stretch, keep proportion, fill / crop, pad)
- `condition` (COMBO): When to resize (always, downscale if bigger, upscale if smaller, if different)
- `multiple_of` (INT): Round dimensions to multiple of this value (0 = disabled)
- `blend_frames` (INT, optional): Number of transition frames for smooth blending
- `loop_additional` (BOOLEAN, optional): Loop additional frames if source is longer
- `memory_efficient` (BOOLEAN, optional): Process in chunks to save memory

**Outputs**:
- `extended_video` (IMAGE): Extended video sequence

**Use Cases**:
- Extend videos by adding intro/outro sequences
- Stitch multiple video clips together
- Add repeated frames for timing adjustments
- Create smooth transitions between sequences

### Frame Replacer ✂️

**Category**: `image/video`

**Inputs**:
- `source_video` (IMAGE): Source video sequence to modify
- `replacement_frames` (IMAGE): Frames to use as replacements
- `target_frame` (INT): Starting frame index for replacement (0-based)
- `replace_count` (INT): Number of frames to replace
- `width` (INT): Target width for frame resizing (64-8192, step 8)
- `height` (INT): Target height for frame resizing (64-8192, step 8)
- `interpolation` (COMBO): Interpolation method (nearest, bilinear, bicubic, area, lanczos)
- `method` (COMBO): Resizing method (stretch, keep proportion, fill / crop, pad)
- `condition` (COMBO): When to resize (always, downscale if bigger, upscale if smaller, if different)
- `multiple_of` (INT): Round dimensions to multiple of this value (0 = disabled)
- `blend_edges` (BOOLEAN, optional): Smooth blend at replacement boundaries
- `blend_strength` (FLOAT, optional): Strength of edge blending (0.0-1.0)
- `loop_replacement` (BOOLEAN, optional): Loop replacement frames if needed
- `preserve_length` (BOOLEAN, optional): Maintain original video length
- `memory_efficient` (BOOLEAN, optional): Process in chunks to save memory

**Outputs**:
- `modified_video` (IMAGE): Modified video sequence

**Use Cases**:
- Replace corrupted or unwanted frames
- Insert new content at specific timestamps
- Fix specific frames in a sequence
- Create frame-accurate edits

### Frame Extender Advanced 🎭

**Category**: `image/video`

**Inputs**:
- All Frame Extender inputs plus:
- `blend_mode` (COMBO): Blending mode (none, linear, ease_in, ease_out, ease_in_out, crossfade, dissolve, overlay)
- `blend_frames` (INT): Number of transition frames (0-100)
- `blend_strength` (FLOAT): Strength of blending effect (0.0-2.0)
- `transition_curve` (COMBO): Transition curve (linear, smooth, sharp, bounce, elastic)
- `reverse_additional` (BOOLEAN, optional): Reverse additional frames before adding
- `fade_edges` (BOOLEAN, optional): Apply fade effect at sequence edges

**Outputs**:
- `extended_video` (IMAGE): Extended video sequence with advanced effects

**Blend Modes**:
- `none`: Simple concatenation (same as basic Frame Extender)
- `linear`: Linear interpolation between frames
- `ease_in/out`: Smooth acceleration/deceleration curves
- `crossfade`: Gamma-corrected smooth transition
- `dissolve`: Random pixel-based transition effect
- `overlay`: Photoshop-style overlay blending

**Transition Curves**:
- `linear`: Constant rate transition
- `smooth`: Smoothstep curve (ease in/out)
- `sharp`: Cubic acceleration
- `bounce`: Bouncing effect
- `elastic`: Oscillating transition

**Use Cases**:
- Professional video transitions
- Creative blend effects
- Artistic video montages
- Smooth scene transitions
- Complex video compositions

## Technical Details

### Image Tensor Format
- **Format**: [Batch, Height, Width, Channels] (ComfyUI standard)
- **Data Type**: float32 (0.0-1.0 range)
- **Channels**: Supports RGB (3) and RGBA (4) channels
- **Batch Dimension**: Represents frame sequence

### Memory Optimization
- **Chunk Processing**: Large videos processed in 32-frame chunks
- **Automatic Resizing**: Efficient tensor operations for resolution matching
- **Memory Monitoring**: Automatic fallback to chunk processing for large sequences

### Error Handling
- **Input Validation**: Comprehensive tensor format checking
- **Graceful Degradation**: Automatic parameter adjustment for edge cases
- **Clear Messages**: Detailed error reporting for troubleshooting

## Examples

### Basic Frame Extension
```
LoadImage -> FrameExtender -> SaveImage
             ^
             additional_frames (LoadImage)
```

### Precise Frame Replacement
```
LoadImage -> FrameReplacer -> SaveImage
             ^
             replacement_frames (LoadImage)
```

### Advanced Workflow with Blending
```
LoadImage -> FrameExtender (blend_frames=5) -> FrameReplacer (blend_edges=True) -> SaveImage
```

### Professional Transitions with Advanced Node
```
LoadImage -> FrameExtenderAdvanced (blend_mode="crossfade", blend_frames=10, transition_curve="smooth") -> SaveImage
             ^
             additional_frames (LoadImage)
```

### Creative Effects Workflow
```
LoadImage -> FrameExtenderAdvanced (blend_mode="dissolve", fade_edges=True, reverse_additional=True) -> SaveImage
```

## Performance Tips

1. **Memory Efficiency**: Enable `memory_efficient` for videos >100 frames
2. **Resolution Matching**: Use `bilinear` for best quality/speed balance
3. **Blending**: Use moderate blend frame counts (3-10) for smooth transitions
4. **Batch Processing**: Process multiple short clips rather than one very long clip

## Troubleshooting

### Common Issues

**"Invalid tensor format"**
- Ensure input is IMAGE type from ComfyUI nodes
- Check tensor dimensions are [B, H, W, C]

**"Out of memory"**
- Enable `memory_efficient` option
- Reduce video length or resolution
- Process in smaller batches

**"Frame index out of bounds"**
- Check `target_frame` is within video length
- Verify `replace_count` doesn't exceed available frames

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.

## Version History

- **v1.0.0**: Initial release
  - FrameExtender: Flexible frame extension with smart blending
  - FrameExtenderAdvanced: Professional blend modes and transition curves
  - FrameReplacer: Precise frame replacement with edge blending
  - Advanced resizing with multiple methods and conditions
  - Memory-efficient processing for large video sequences
  - Comprehensive error handling and validation
