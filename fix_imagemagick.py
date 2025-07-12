#!/usr/bin/env python3
"""
Script to configure MoviePy to find ImageMagick on Windows.
"""

import os
import sys
from moviepy.config import change_settings

# Common ImageMagick installation paths on Windows
possible_paths = [
    r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
    r"C:\Program Files\ImageMagick-7.1.1-Q16\magick.exe",
    r"C:\Program Files\ImageMagick-7.1.0-Q16-HDRI\magick.exe",
    r"C:\Program Files\ImageMagick-7.1.0-Q16\magick.exe",
    r"C:\Program Files\ImageMagick-7.0.11-Q16-HDRI\magick.exe",
    r"C:\Program Files\ImageMagick-7.0.11-Q16\magick.exe",
    r"C:\Program Files\ImageMagick-7.0.10-Q16-HDRI\magick.exe",
    r"C:\Program Files\ImageMagick-7.0.10-Q16\magick.exe",
]

print("Searching for ImageMagick installation...")

# First, try to find ImageMagick in PATH
try:
    import subprocess
    result = subprocess.run(['magick', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Found ImageMagick in PATH")
        # Get the full path
        result = subprocess.run(['where', 'magick'], capture_output=True, text=True)
        if result.returncode == 0:
            magick_path = result.stdout.strip().split('\n')[0]
            print(f"ImageMagick path: {magick_path}")
            
            # Configure MoviePy
            change_settings({"IMAGEMAGICK_BINARY": magick_path})
            print("✓ MoviePy configured to use ImageMagick from PATH")
        else:
            print("Could not get full path of magick command")
    else:
        print("ImageMagick not found in PATH")
except Exception as e:
    print(f"Error checking PATH: {e}")

# If not in PATH, try common installation locations
if not any(os.path.exists(path) for path in possible_paths):
    print("Searching common installation directories...")
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ Found ImageMagick at: {path}")
            change_settings({"IMAGEMAGICK_BINARY": path})
            print("✓ MoviePy configured successfully!")
            break
    else:
        print("✗ ImageMagick not found in common locations")
        print("Please check if ImageMagick is installed correctly")
        sys.exit(1)

# Test the configuration
print("\nTesting MoviePy configuration...")
try:
    from moviepy.editor import TextClip
    text = TextClip("TEST", fontsize=20, color='black')
    print("✓ TextClip created successfully!")
    text.close()
    print("✓ MoviePy is now properly configured!")
except Exception as e:
    print(f"✗ Still having issues: {e}")
    print("You may need to restart your Python environment") 