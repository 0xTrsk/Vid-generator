#!/usr/bin/env python3
"""
Launcher script for Advanced Subtitle Generator
Checks dependencies and launches the main application
"""

import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'whisper',
        'pydub', 
        'pysrt',
        'moviepy',
        'nltk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def check_whisperx():
    """Check if WhisperX is available (optional)"""
    try:
        import whisperx
        print("✅ WhisperX is available (enhanced timestamping)")
        return True
    except ImportError:
        print("⚠️  WhisperX not available (will use regular Whisper)")
        print("Install with: pip install git+https://github.com/m-bain/whisperX.git")
        return False

def check_external_tools():
    """Check if external tools are available"""
    print("\nChecking external tools...")
    
    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
        else:
            print("❌ FFmpeg not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ FFmpeg not found")
        print("Please install FFmpeg from: https://ffmpeg.org/download.html")
        return False
    
    # Check ImageMagick
    try:
        result = subprocess.run(['magick', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ ImageMagick is available")
        else:
            print("❌ ImageMagick not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ ImageMagick not found")
        print("Please install ImageMagick from: https://imagemagick.org/script/download.php")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("=" * 50)
    print("Advanced Subtitle Generator Launcher")
    print("=" * 50)
    print()
    
    # Check Python version
    if not check_python_version():
        input("\nPress Enter to exit...")
        return
    
    print()
    
    # Check dependencies
    if not check_dependencies():
        input("\nPress Enter to exit...")
        return
    
    print()
    
    # Check WhisperX (optional)
    check_whisperx()
    
    print()
    
    # Check external tools
    if not check_external_tools():
        print("\n⚠️  Some external tools are missing, but you can still try to run the application.")
        print("Video generation may not work without FFmpeg and ImageMagick.")
    
    print("\n" + "=" * 50)
    print("Launching Advanced Subtitle Generator...")
    print("=" * 50)
    print()
    
    try:
        # Import and run the main application
        from advanced_subtitle_generator import main as app_main
        app_main()
    except ImportError as e:
        print(f"❌ Error importing main application: {e}")
        print("Make sure advanced_subtitle_generator.py is in the same directory")
        input("\nPress Enter to exit...")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main() 