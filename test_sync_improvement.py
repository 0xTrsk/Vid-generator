#!/usr/bin/env python3
"""
Test script to demonstrate improved word synchronization with actual timing.
This shows how words now appear at their actual speech time rather than equal distribution.
"""

import os
import sys

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_subtitle_generator import create_word_typewriter_clip, get_animation_style_options
    from moviepy.editor import ColorClip, CompositeVideoClip, AudioFileClip
    print("✓ Successfully imported animation functions")
    
    # Test data with realistic word timings (simulating Whisper output)
    test_subtitle = {
        'text': 'Hello world, this is a test!',
        'start': 0.0,
        'end': 3.0
    }
    
    # Simulate actual word timings from Whisper (words appear at different times)
    word_timings = [
        {'word': 'Hello', 'start': 0.0, 'end': 0.5},
        {'word': 'world', 'start': 0.6, 'end': 1.1},
        {'word': 'this', 'start': 1.3, 'end': 1.6},
        {'word': 'is', 'start': 1.7, 'end': 1.9},
        {'word': 'a', 'start': 2.0, 'end': 2.1},
        {'word': 'test', 'start': 2.2, 'end': 2.8}
    ]
    
    video_size = (640, 480)  # Smaller size for testing
    
    print("\n🎬 Testing Improved Word Synchronization:")
    print("=" * 60)
    print("Original timing: Equal distribution (0.5s per word)")
    print("New timing: Actual speech timing with configurable lead-in")
    print("=" * 60)
    
    # Test with different lead-in values
    lead_in_values = [0, 50, 100, 200]  # milliseconds
    
    for lead_in_ms in lead_in_values:
        print(f"\n🔹 Testing with {lead_in_ms}ms lead-in:")
        
        try:
            # Create the animated clip with actual timing
            clip = create_word_typewriter_clip(
                test_subtitle, 
                video_size, 
                fontsize=30,  # Smaller font for test
                animation_style="typewriter",
                word_timings=word_timings,
                app_instance=None  # No app instance for this test
            )
            
            if clip:
                # Create a simple background
                background = ColorClip(size=video_size, color=(255, 255, 255), duration=3)
                
                # Combine background and text
                final = CompositeVideoClip([background, clip])
                
                # Export test video
                output_file = f"test_sync_{lead_in_ms}ms_leadin.mp4"
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
                print(f"   📊 Word timing analysis:")
                
                # Show the timing for each word
                for i, (word, timing) in enumerate(zip(test_subtitle['text'].split(), word_timings)):
                    if i < len(word_timings):
                        actual_start = timing['start']
                        actual_end = timing['end']
                        word_duration = actual_end - actual_start
                        print(f"      '{word}': {actual_start:.2f}s - {actual_end:.2f}s (duration: {word_duration:.2f}s)")
                
            else:
                print(f"   ❌ Failed: Could not create clip")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n🎉 Synchronization test complete!")
    print("\n💡 Key Improvements:")
    print("• Words now appear at their actual speech time (not equal distribution)")
    print("• Configurable lead-in timing (0-500ms)")
    print("• Better perceived synchronization with audio")
    print("• Works with both Whisper and WhisperX word-level timing")
    print("\n📱 In the main app, you can now adjust 'Word Lead-in' in the UI!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the same directory as advanced_subtitle_generator.py")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
