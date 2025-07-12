#!/usr/bin/env python3
"""
Simple test script to verify MoviePy text rendering works.
Run this to test if MoviePy can create text overlays.
"""

try:
    from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
    print("✓ MoviePy imported successfully")
    
    # Create a simple test video
    print("Creating test video...")
    
    # White background
    video = ColorClip(size=(640, 480), color=(255, 255, 255), duration=5)
    
    # Simple text clip
    text = TextClip("TEST TEXT", fontsize=40, color='black')
    text = text.set_start(1.0).set_end(4.0)
    text = text.set_position(('center', 'center'))
    
    # Combine
    final = CompositeVideoClip([video, text])
    
    # Export
    print("Exporting test video...")
    final.write_videofile("test_output.mp4", fps=24, verbose=False, logger=None)
    
    print("✓ Test video created successfully!")
    print("Check 'test_output.mp4' - you should see 'TEST TEXT' in the center from 1-4 seconds")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install MoviePy: pip install moviepy")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc() 