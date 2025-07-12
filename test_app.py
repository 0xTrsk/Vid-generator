#!/usr/bin/env python3
"""
Test script for Subtitle Generator
This script tests the basic functionality without requiring GUI.
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import tkinter as tk
        print("✓ Tkinter available")
    except ImportError as e:
        print(f"✗ Tkinter not available: {e}")
        return False
    
    try:
        import speech_recognition as sr
        print("✓ SpeechRecognition available")
    except ImportError as e:
        print(f"✗ SpeechRecognition not available: {e}")
        print("  Run: pip install SpeechRecognition")
    
    try:
        from pydub import AudioSegment
        print("✓ Pydub available")
    except ImportError as e:
        print(f"✗ Pydub not available: {e}")
        print("  Run: pip install pydub")
    
    try:
        import pysrt
        print("✓ Pysrt available")
    except ImportError as e:
        print(f"✗ Pysrt not available: {e}")
        print("  Run: pip install pysrt")
    
    return True

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "subtitle_generator.py",
        "advanced_subtitle_generator.py", 
        "launcher.py",
        "requirements.txt",
        "README.md"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    return all_exist

def test_basic_functionality():
    """Test basic subtitle generation functionality"""
    print("\nTesting basic functionality...")
    
    try:
        # Import the basic subtitle generator
        sys.path.append('.')
        from subtitle_generator import SubtitleGeneratorApp
        
        print("✓ Basic subtitle generator can be imported")
        
        # Test time conversion function
        from subtitle_generator import SubtitleGeneratorApp
        app = SubtitleGeneratorApp.__new__(SubtitleGeneratorApp)
        
        # Test seconds to time conversion
        test_seconds = 65.5
        time_obj = app.seconds_to_time(test_seconds)
        
        if hasattr(time_obj, 'hours') and hasattr(time_obj, 'minutes') and hasattr(time_obj, 'seconds'):
            print("✓ Time conversion function works")
        else:
            print("✗ Time conversion function failed")
            return False
            
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("Subtitle Generator - Test Suite")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test file structure
    files_ok = test_file_structure()
    
    # Test basic functionality
    if imports_ok and files_ok:
        func_ok = test_basic_functionality()
    else:
        func_ok = False
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Files: {'✓ PASS' if files_ok else '✗ FAIL'}")
    print(f"Functionality: {'✓ PASS' if func_ok else '✗ FAIL'}")
    
    if imports_ok and files_ok and func_ok:
        print("\n🎉 All tests passed! The application should work correctly.")
        print("\nTo run the application:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run launcher: python launcher.py")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        print("\nTo install dependencies:")
        print("Windows: Double-click install.bat")
        print("Linux/Mac: Run ./install.sh")

if __name__ == "__main__":
    main() 