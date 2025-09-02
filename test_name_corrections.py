#!/usr/bin/env python3
"""
Test script to demonstrate enhanced name correction features.
This shows how names and proper nouns are now better handled in subtitles.
"""

import os
import sys

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the correction functions
    from advanced_subtitle_generator import (
        extract_names_from_script, 
        create_script_based_corrections,
        enhanced_correct_known_errors,
        CORRECTIONS,
        PHRASE_CORRECTIONS
    )
    print("✓ Successfully imported name correction functions")
    
    # Test data
    print("\n🎬 Testing Enhanced Name Correction Features:")
    print("=" * 60)
    
    # Test 1: Extract names from script
    print("\n🔹 Test 1: Name Extraction from Script")
    print("-" * 40)
    
    sample_script = """
    Dr. John Smith is a professor at Harvard University.
    He works with Dr. Sarah Johnson and Mr. Michael Brown.
    They are developing a new AI system called "Project Phoenix".
    The team includes Emma Wilson, David Lee, and Lisa Chen.
    They use Python, JavaScript, and HTML for development.
    """
    
    extracted_names = extract_names_from_script(sample_script)
    print(f"Extracted {len(extracted_names)} potential names/entities:")
    for name_lower, name_original in extracted_names.items():
        print(f"  '{name_lower}' → '{name_original}'")
    
    # Test 2: Script-based corrections
    print("\n🔹 Test 2: Script-Based Corrections")
    print("-" * 40)
    
    # Simulate transcription with errors
    transcription_with_errors = """
    doctor john smith is a professor at harvard university
    he works with doctor sarah johnson and mister michael brown
    they are developing a new ai system called project phoenix
    the team includes emma wilson david lee and lisa chen
    they use python javascript and html for development
    """
    
    script_corrections = create_script_based_corrections(sample_script, transcription_with_errors)
    print(f"Generated {len(script_corrections)} script-based corrections:")
    for wrong, correct in script_corrections.items():
        print(f"  '{wrong}' → '{correct}'")
    
    # Test 3: Enhanced error correction
    print("\n🔹 Test 3: Enhanced Error Correction")
    print("-" * 40)
    
    # Sample subtitle lines with errors
    subtitle_lines_with_errors = [
        {'text': 'doctor john smith is a professor', 'start': 0.0, 'end': 3.0},
        {'text': 'at harvard university', 'start': 3.0, 'end': 5.0},
        {'text': 'he works with doctor sarah johnson', 'start': 5.0, 'end': 8.0},
        {'text': 'and mister michael brown', 'start': 8.0, 'end': 10.0},
        {'text': 'they use python javascript and html', 'start': 10.0, 'end': 13.0}
    ]
    
    print("Original subtitles with errors:")
    for i, sub in enumerate(subtitle_lines_with_errors, 1):
        print(f"  {i}. '{sub['text']}'")
    
    # Apply enhanced corrections
    corrected_subtitles = enhanced_correct_known_errors(
        subtitle_lines_with_errors, 
        CORRECTIONS, 
        script_corrections
    )
    
    print("\nCorrected subtitles:")
    for i, sub in enumerate(corrected_subtitles, 1):
        print(f"  {i}. '{sub['text']}'")
    
    # Test 4: Built-in corrections
    print("\n🔹 Test 4: Built-in Corrections")
    print("-" * 40)
    
    print(f"Built-in corrections ({len(CORRECTIONS)} total):")
    for wrong, correct in list(CORRECTIONS.items())[:10]:  # Show first 10
        print(f"  '{wrong}' → '{correct}'")
    
    if len(CORRECTIONS) > 10:
        print(f"  ... and {len(CORRECTIONS) - 10} more")
    
    print(f"\nPhrase corrections ({len(PHRASE_CORRECTIONS)} total):")
    for wrong, correct in PHRASE_CORRECTIONS.items():
        print(f"  '{wrong}' → '{correct}'")
    
    # Test 5: Real-world example
    print("\n🔹 Test 5: Real-World Example")
    print("-" * 40)
    
    real_script = """
    Dr. Mitja Machtig is presenting at the Move UK conference.
    He will discuss the SCR30 project with Professor Brown.
    The event is sponsored by Microsoft and Google.
    """
    
    real_transcription = """
    doctor mitja machtig is presenting at the move uk conference
    he will discuss the scr thirty project with professor brown
    the event is sponsored by microsoft and google
    """
    
    real_corrections = create_script_based_corrections(real_script, real_transcription)
    print(f"Real-world corrections: {len(real_corrections)} found")
    for wrong, correct in real_corrections.items():
        print(f"  '{wrong}' → '{correct}'")
    
    print("\n🎉 Name correction test complete!")
    print("\n💡 Key Features:")
    print("• Automatic name detection from scripts")
    print("• Script-based transcription correction")
    print("• Built-in corrections for common names/terms")
    print("• Fuzzy matching for better accuracy")
    print("• Case-preserving corrections")
    print("\n📱 In the main app, you can now:")
    print("• Add custom name corrections in the UI")
    print("• See automatic script-based corrections")
    print("• Get better subtitle accuracy for names")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the same directory as advanced_subtitle_generator.py")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
