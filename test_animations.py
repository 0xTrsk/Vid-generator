#!/usr/bin/env python3
"""
Test script to demonstrate different animation styles for subtitles.
Run this to see how each animation style looks.
"""

import os
import sys

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_subtitle_generator import create_word_typewriter_clip, get_animation_style_options
    from moviepy.editor import ColorClip, CompositeVideoClip, concatenate_videoclips
    print("✓ Successfully imported animation functions")
    
    # Test data
    test_subtitle = {
        'text': 'Hello world, this is a test of different animation styles!',
        'start': 0.0,
        'end': 3.0
    }
    
    video_size = (640, 480)  # Smaller size for testing
    
    print("\n🎬 Testing Animation Styles:")
    print("=" * 50)
    
    # Get available styles
    styles = get_animation_style_options()
    
    for style_name, description in styles.items():
        print(f"\n🔹 Testing: {style_name}")
        print(f"   Description: {description}")
        
        try:
            # Create the animated clip
            clip = create_word_typewriter_clip(
                test_subtitle, 
                video_size, 
                fontsize=30,  # Smaller font for test
                animation_style=style_name
            )
            
            if clip:
                # Create a simple background
                background = ColorClip(size=video_size, color=(255, 255, 255), duration=3)
                
                # Combine background and text
                final = CompositeVideoClip([background, clip])
                
                # Export test video
                output_file = f"test_animation_{style_name}.mp4"
                print(f"   ✓ Creating: {output_file}")
                
                final.write_videofile(
                    output_file, 
                    fps=24, 
                    verbose=False, 
                    logger=None,
                    codec='libx264',
                    audio_codec='aac'
                )
                
                print(f"   ✅ Success: {output_file}")
                
            else:
                print(f"   ❌ Failed: Could not create clip")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n🎉 Animation test complete!")
    print("Check the generated MP4 files to see each animation style in action.")
    print("\n💡 Tip: You can now use these animation styles in the main application!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the same directory as advanced_subtitle_generator.py")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
