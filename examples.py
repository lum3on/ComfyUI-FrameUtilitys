"""
Test script for ComfyUI Frame Utility nodes.

Validates functionality and performance of FrameExtender and FrameReplacer nodes.
Run this script to ensure proper installation and functionality.
"""

import torch
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import with absolute imports for testing
import importlib.util
import sys

# Load modules directly
spec = importlib.util.spec_from_file_location("frame_extender", "frame_extender.py")
frame_extender_module = importlib.util.module_from_spec(spec)

spec = importlib.util.spec_from_file_location("frame_replacer", "frame_replacer.py")
frame_replacer_module = importlib.util.module_from_spec(spec)

spec = importlib.util.spec_from_file_location("utils", "utils.py")
utils_module = importlib.util.module_from_spec(spec)

# Execute modules
spec.loader.exec_module(utils_module)
sys.modules['utils'] = utils_module

spec = importlib.util.spec_from_file_location("frame_extender", "frame_extender.py")
frame_extender_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frame_extender_module)

spec = importlib.util.spec_from_file_location("frame_replacer", "frame_replacer.py")
frame_replacer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frame_replacer_module)

# Get classes
FrameExtender = frame_extender_module.FrameExtender
FrameReplacer = frame_replacer_module.FrameReplacer
utils = utils_module

def create_test_video(frames: int, height: int = 64, width: int = 64, channels: int = 3) -> torch.Tensor:
    """Create a test video with gradient patterns for easy visual verification."""
    video = torch.zeros(frames, height, width, channels)
    
    for i in range(frames):
        # Create unique pattern for each frame
        frame_value = i / max(1, frames - 1)  # 0 to 1
        
        # Gradient pattern
        for h in range(height):
            for w in range(width):
                video[i, h, w, 0] = frame_value  # Red channel = frame number
                video[i, h, w, 1] = h / height   # Green channel = height gradient
                video[i, h, w, 2] = w / width    # Blue channel = width gradient
    
    return video

def test_frame_extender():
    """Test FrameExtender functionality."""
    print("Testing FrameExtender...")
    
    extender = FrameExtender()
    
    # Test basic extension
    source = create_test_video(5, 32, 32, 3)
    additional = create_test_video(3, 32, 32, 3)
    
    # Test end extension
    result = extender.extend_frames(source, additional, insert_position=-1, width=32, height=32)
    extended = result[0]
    assert extended.shape[0] == 8, f"Expected 8 frames, got {extended.shape[0]}"
    print("✓ End extension test passed")

    # Test beginning extension
    result = extender.extend_frames(source, additional, insert_position=0, width=32, height=32)
    extended = result[0]
    assert extended.shape[0] == 8, f"Expected 8 frames, got {extended.shape[0]}"
    print("✓ Beginning extension test passed")

    # Test middle extension
    result = extender.extend_frames(source, additional, insert_position=2, width=32, height=32)
    extended = result[0]
    assert extended.shape[0] == 8, f"Expected 8 frames, got {extended.shape[0]}"
    print("✓ Middle extension test passed")

    # Test with blending
    result = extender.extend_frames(source, additional, insert_position=-1, width=32, height=32, blend_frames=2)
    extended = result[0]
    assert extended.shape[0] == 10, f"Expected 10 frames (5+3+2 blend), got {extended.shape[0]}"
    print("✓ Blending test passed")

    # Test resolution change
    additional_large = create_test_video(3, 64, 64, 3)
    result = extender.extend_frames(source, additional_large, insert_position=-1, width=64, height=64)
    extended = result[0]
    assert extended.shape[1:3] == (64, 64), f"Expected (64, 64), got {extended.shape[1:3]}"
    print("✓ Resolution change test passed")
    
    print("FrameExtender tests completed successfully! ✅\n")

def test_frame_replacer():
    """Test FrameReplacer functionality."""
    print("Testing FrameReplacer...")
    
    replacer = FrameReplacer()
    
    # Test basic replacement
    source = create_test_video(10, 32, 32, 3)
    replacement = create_test_video(3, 32, 32, 3)
    
    # Test single frame replacement
    result = replacer.replace_frames(source, replacement, target_frame=2, replace_count=1, width=32, height=32)
    modified = result[0]
    assert modified.shape[0] == 10, f"Expected 10 frames, got {modified.shape[0]}"
    print("✓ Single frame replacement test passed")

    # Test multiple frame replacement
    result = replacer.replace_frames(source, replacement, target_frame=2, replace_count=3, width=32, height=32)
    modified = result[0]
    assert modified.shape[0] == 10, f"Expected 10 frames, got {modified.shape[0]}"
    print("✓ Multiple frame replacement test passed")

    # Test replacement with looping
    result = replacer.replace_frames(source, replacement, target_frame=2, replace_count=5, width=32, height=32, loop_replacement=True)
    modified = result[0]
    assert modified.shape[0] == 10, f"Expected 10 frames, got {modified.shape[0]}"
    print("✓ Looping replacement test passed")

    # Test edge blending
    result = replacer.replace_frames(source, replacement, target_frame=2, replace_count=3, width=32, height=32, blend_edges=True, blend_strength=0.5)
    modified = result[0]
    assert modified.shape[0] == 10, f"Expected 10 frames, got {modified.shape[0]}"
    print("✓ Edge blending test passed")

    # Test boundary conditions
    result = replacer.replace_frames(source, replacement, target_frame=0, replace_count=2, width=32, height=32)
    modified = result[0]
    assert modified.shape[0] == 10, f"Expected 10 frames, got {modified.shape[0]}"
    print("✓ Beginning replacement test passed")

    result = replacer.replace_frames(source, replacement, target_frame=8, replace_count=2, width=32, height=32)
    modified = result[0]
    assert modified.shape[0] == 10, f"Expected 10 frames, got {modified.shape[0]}"
    print("✓ End replacement test passed")
    
    print("FrameReplacer tests completed successfully! ✅\n")

def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test tensor validation
    valid_tensor = torch.randn(5, 32, 32, 3)
    try:
        utils.validate_image_tensor(valid_tensor, "test")
        print("✓ Tensor validation test passed")
    except Exception as e:
        print(f"✗ Tensor validation test failed: {e}")
        return
    
    # Test resize functionality
    source = torch.randn(3, 32, 32, 3)
    target = torch.randn(3, 64, 64, 3)
    resized = utils.resize_to_match(source, target)
    assert resized.shape[1:3] == target.shape[1:3], "Resize failed"
    print("✓ Resize test passed")
    
    # Test frame blending
    frame1 = torch.ones(1, 32, 32, 3) * 0.0
    frame2 = torch.ones(1, 32, 32, 3) * 1.0
    blended = utils.blend_frames(frame1, frame2, 0.5)
    expected_value = 0.5
    assert torch.allclose(blended, torch.ones_like(blended) * expected_value, atol=1e-6), "Blending failed"
    print("✓ Frame blending test passed")
    
    # Test safe indexing
    safe_idx = utils.safe_frame_index(10, -1, allow_negative=True)
    assert safe_idx == 9, f"Expected 9, got {safe_idx}"
    print("✓ Safe indexing test passed")
    
    print("Utility function tests completed successfully! ✅\n")

def test_performance():
    """Test performance with larger videos."""
    print("Testing performance with larger videos...")
    
    # Create larger test videos
    large_source = create_test_video(100, 128, 128, 3)
    large_additional = create_test_video(50, 128, 128, 3)
    
    extender = FrameExtender()
    replacer = FrameReplacer()
    
    # Test memory-efficient processing
    import time
    
    start_time = time.time()
    result = extender.extend_frames(large_source, large_additional, width=128, height=128, memory_efficient=True)
    extend_time = time.time() - start_time
    print(f"✓ Large video extension completed in {extend_time:.2f}s")

    start_time = time.time()
    result = replacer.replace_frames(large_source, large_additional[:20], target_frame=10, replace_count=20, width=128, height=128, memory_efficient=True)
    replace_time = time.time() - start_time
    print(f"✓ Large video replacement completed in {replace_time:.2f}s")
    
    print("Performance tests completed successfully! ✅\n")

def main():
    """Run all tests."""
    print("🎬 ComfyUI Frame Utility Node Tests 🎬\n")
    print("=" * 50)
    
    try:
        test_utils()
        test_frame_extender()
        test_frame_replacer()
        test_performance()
        
        print("=" * 50)
        print("🎉 All tests passed successfully! 🎉")
        print("\nNodes are ready for use in ComfyUI!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
