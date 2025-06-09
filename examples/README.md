# ComfyUI Frame Utility - Examples

This directory contains example workflows and usage demonstrations for the ComfyUI Frame Utility nodes.

## Files

### Workflow Examples
- `frameXtnd&framereplce_workflow.json` - Combined workflow showing FrameExtender and FrameReplacer usage

### Python Examples
- `../examples.py` - Python script demonstrating basic usage of all nodes

## How to Use Workflow Files

1. Open ComfyUI in your browser
2. Click "Load" button
3. Select one of the `.json` files from this directory
4. The workflow will load with pre-configured nodes
5. Connect your input images/videos
6. Queue the workflow to see the results

## Basic Usage Patterns

### Frame Extension
```
LoadImage -> FrameExtender -> SaveImage
             ^
             additional_frames (LoadImage)
```

### Frame Replacement
```
LoadImage -> FrameReplacer -> SaveImage
             ^
             replacement_frames (LoadImage)
```

### Advanced Effects
```
LoadImage -> FrameExtenderAdvanced -> SaveImage
             ^
             additional_frames (LoadImage)
```

## Tips

- Use IMAGE outputs from other ComfyUI nodes as inputs
- All nodes support batch processing
- Enable memory_efficient for large videos
- Experiment with different blend modes and parameters
- Check the console for detailed processing information

## Creating Your Own Workflows

1. Add the Frame Utility nodes to your workflow
2. Connect IMAGE tensors from other nodes
3. Configure parameters as needed
4. Save your workflow for reuse

For more detailed documentation, see the main README.md file.
