#!/bin/bash

echo "========================================"
echo "Advanced Subtitle Generator Installer"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo
echo "Optional: Install WhisperX for enhanced timestamping..."
echo "(This may take a few minutes)"
pip3 install git+https://github.com/m-bain/whisperX.git

echo
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Install FFmpeg: sudo apt install ffmpeg (Ubuntu/Debian)"
echo "   or: brew install ffmpeg (macOS)"
echo "2. Install ImageMagick: sudo apt install imagemagick (Ubuntu/Debian)"
echo "   or: brew install imagemagick (macOS)"
echo "3. Run: python3 advanced_subtitle_generator.py"
echo 