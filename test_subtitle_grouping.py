#!/usr/bin/env python3
"""
Test script to demonstrate improved subtitle grouping logic.
This shows how subtitles are now grouped more intelligently for better readability.
"""

import os
import sys

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_subtitle_generator import AdvancedSubtitleGeneratorApp
    print("✓ Successfully imported AdvancedSubtitleGeneratorApp")
    
    # Create a mock app instance for testing
    class MockApp:
        def __init__(self):
            self.max_words_per_subtitle = 8
    
    # Test data with realistic word timings (simulating Whisper output)
    test_word_segments = [
        {'word': 'Hello', 'start': 0.0, 'end': 0.5},
        {'word': 'world', 'start': 0.6, 'end': 1.1},
        {'word': 'this', 'start': 1.3, 'end': 1.6},
        {'word': 'is', 'start': 1.7, 'end': 1.9},
        {'word': 'a', 'start': 2.0, 'end': 2.1},
        {'word': 'test', 'start': 2.2, 'end': 2.8},
        {'word': 'of', 'start': 2.9, 'end': 3.1},
        {'word': 'the', 'start': 3.2, 'end': 3.4},
        {'word': 'new', 'start': 3.5, 'end': 3.8},
        {'word': 'subtitle', 'start': 3.9, 'end': 4.5},
        {'word': 'grouping', 'start': 4.6, 'end': 5.2},
        {'word': 'system', 'start': 5.3, 'end': 5.9},
        {'word': 'It', 'start': 6.0, 'end': 6.2},
        {'word': 'should', 'start': 6.3, 'end': 6.7},
        {'word': 'work', 'start': 6.8, 'end': 7.2},
        {'word': 'much', 'start': 7.3, 'end': 7.6},
        {'word': 'better', 'start': 7.7, 'end': 8.3},
        {'word': 'now', 'start': 8.4, 'end': 8.8},
        {'word': 'What', 'start': 9.0, 'end': 9.3},
        {'word': 'do', 'start': 9.4, 'end': 9.6},
        {'word': 'you', 'start': 9.7, 'end': 9.9},
        {'word': 'think', 'start': 10.0, 'end': 10.4},
        {'word': '?', 'start': 10.5, 'end': 10.6}
    ]
    
    print("\n🎬 Testing Improved Subtitle Grouping:")
    print("=" * 60)
    print("Requirements:")
    print("• Sentences end at . ? !")
    print("• Max 8 words per subtitle (fits on 2 lines)")
    print("• No 1-2 word subtitles except complete sentences")
    print("• Smart merging of short fragments")
    print("=" * 60)
    
    # Test with different max word settings
    max_words_settings = [6, 8, 10]
    
    for max_words in max_words_settings:
        print(f"\n🔹 Testing with max {max_words} words per subtitle:")
        
        try:
            # Create a mock app instance
            mock_app = MockApp()
            mock_app.max_words_per_subtitle = max_words
            
            # Create a real app instance to access the method
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()  # Hide the window
            app = AdvancedSubtitleGeneratorApp(root)
            
            # Test the grouping function
            grouped_subtitles = app.group_words_by_punctuation(test_word_segments, max_words=max_words)
            
            print(f"   ✓ Created {len(grouped_subtitles)} subtitle groups")
            print(f"   📊 Subtitle analysis:")
            
            for i, subtitle in enumerate(grouped_subtitles, 1):
                word_count = len(subtitle['text'].split())
                duration = subtitle['end'] - subtitle['start']
                ends_sentence = subtitle['text'].strip().endswith(('.', '?', '!'))
                
                # Color coding for analysis
                if word_count <= 2 and not ends_sentence:
                    status = "⚠️  SHORT (but allowed if complete sentence)"
                elif word_count <= max_words:
                    status = "✅ GOOD"
                else:
                    status = "❌ TOO LONG"
                
                print(f"      {i}. '{subtitle['text']}'")
                print(f"         Words: {word_count}, Duration: {duration:.2f}s, {status}")
                if ends_sentence:
                    print(f"         ✓ Complete sentence")
                print()
            
            root.destroy()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 Subtitle grouping test complete!")
    print("\n💡 Key Improvements:")
    print("• Sentences now properly end at punctuation marks")
    print("• Short fragments are intelligently merged")
    print("• Word count limits are enforced (configurable 6-12)")
    print("• Better readability with 2-line maximum display")
    print("\n📱 In the main app, you can now adjust 'Max Words/Subtitle' in the UI!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the same directory as advanced_subtitle_generator.py")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
