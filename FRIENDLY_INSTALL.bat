@echo off
echo ========================================
echo    Video Generator - Friend Installer
echo ========================================
echo.
echo This will help you install the video generator app!
echo.

REM Check if Python is installed
echo Checking if Python is installed...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ❌ Python is not installed or not in PATH!
    echo.
    echo Please install Python first:
    echo 1. Go to https://python.org
    echo 2. Download the latest version
    echo 3. Run the installer and CHECK "Add Python to PATH"
    echo 4. Restart this installer
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Python is installed!
)

REM Check if FFmpeg is installed
echo.
echo Checking if FFmpeg is installed...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ❌ FFmpeg is not installed or not in PATH!
    echo.
    echo Please install FFmpeg:
    echo 1. Go to https://ffmpeg.org/download.html
    echo 2. Click "Windows builds"
    echo 3. Download the latest release
    echo 4. Extract to C:\ffmpeg
    echo 5. Restart this installer
    echo.
    pause
    exit /b 1
) else (
    echo ✅ FFmpeg is installed!
)

REM Check if ImageMagick is installed
echo.
echo Checking if ImageMagick is installed...
if not exist "C:\ImageMagick\magick.exe" (
    echo.
    echo ❌ ImageMagick is not properly set up!
    echo.
    echo Please install ImageMagick:
    echo 1. Go to https://imagemagick.org/script/download.php
    echo 2. Download Windows Binary Release
    echo 3. Install with default settings
    echo 4. Copy magick.exe to C:\ImageMagick\
    echo 5. Restart this installer
    echo.
    pause
    exit /b 1
) else (
    echo ✅ ImageMagick is installed!
)

echo.
echo ========================================
echo    Installing Python Dependencies
echo ========================================
echo.

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ❌ Failed to install requirements!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Installing WhisperX (Optional)
echo ========================================
echo.

echo Do you want to install WhisperX for better transcription? (y/n)
set /p choice=
if /i "%choice%"=="y" (
    echo Installing WhisperX...
    pip install git+https://github.com/m-bain/whisperX.git
    if %errorlevel% neq 0 (
        echo.
        echo ⚠️ WhisperX installation failed, but the app will still work!
        echo The app will use regular Whisper instead.
    ) else (
        echo ✅ WhisperX installed successfully!
    )
) else (
    echo Skipping WhisperX installation.
)

echo.
echo ========================================
echo    Installation Complete! 🎉
echo ========================================
echo.
echo ✅ Everything is installed and ready!
echo.
echo To run the app:
echo 1. Type: python launcher.py
echo 2. Or double-click launcher.py
echo.
echo For help, read FRIENDLY_INSTALL_GUIDE.md
echo.
pause 