@echo off
echo ========================================
echo Advanced Subtitle Generator Installer
echo ========================================
echo.

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Optional: Install WhisperX for enhanced timestamping...
echo (This may take a few minutes)
pip install git+https://github.com/m-bain/whisperX.git

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Install FFmpeg from: https://ffmpeg.org/download.html
echo 2. Install ImageMagick from: https://imagemagick.org/script/download.php
echo 3. Run: python advanced_subtitle_generator.py
echo.
pause 