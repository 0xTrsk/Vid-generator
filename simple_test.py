#!/usr/bin/env python3
print("Testing MoviePy TextClip...")

try:
    from moviepy.editor import TextClip
    print("✓ MoviePy imported successfully")
    
    # Try to create a simple text clip
    text = TextClip("TEST", fontsize=20, color='black')
    print("✓ TextClip created successfully!")
    text.close()
    print("✓ TextClip closed successfully!")
    
    print("ImageMagick is working correctly!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc() 