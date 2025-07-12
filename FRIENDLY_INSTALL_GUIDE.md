# 🎬 Video Generator - Easy Installation Guide for Friends

**Hey friends! Here's how to install and use my video generator app in 5 simple steps!**

## 📋 What You Need First

### 1. Python (Required)
- **Download Python**: Go to [python.org](https://python.org)
- **Click "Download Python"** (get the latest version)
- **Install it**: Run the installer and **check "Add Python to PATH"**
- **Test**: Open Command Prompt and type `python --version` (should show a version number)

### 2. FFmpeg (Required)
- **Download**: Go to [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- **Click "Windows builds"** → **Download the latest release**
- **Extract**: Unzip the file to `C:\ffmpeg` (create this folder)
- **Test**: Open Command Prompt and type `ffmpeg -version` (should show version info)

### 3. ImageMagick (Required)
- **Download**: Go to [imagemagick.org/script/download.php](https://imagemagick.org/script/download.php)
- **Click "Windows Binary Release"** → **Download the latest version**
- **Install**: Run the installer with default settings
- **Copy file**: Copy `magick.exe` from your ImageMagick install folder to `C:\ImageMagick\`

## 🚀 Installation Steps

### Step 1: Download the Project
- **Go to**: [Your GitHub repository URL]
- **Click the green "Code" button**
- **Click "Download ZIP"**
- **Extract** the ZIP file to your Desktop

### Step 2: Open Command Prompt
- **Press `Windows + R`**
- **Type `cmd`** and press Enter
- **Navigate to the project folder**:
  ```
  cd Desktop\Video generator
  ```

### Step 3: Install Dependencies
**Option A - Easy Way (Windows):**
- **Double-click** `install.bat` in the project folder

**Option B - Manual Way:**
- In Command Prompt, type:
  ```
  pip install -r requirements.txt
  ```

### Step 4: Install WhisperX (Optional but Recommended)
- In Command Prompt, type:
  ```
  pip install git+https://github.com/m-bain/whisperX.git
  ```

### Step 5: Run the App!
- In Command Prompt, type:
  ```
  python launcher.py
  ```

## 🎯 How to Use

### First Time Setup
1. **Click "Add Pair"** in the app
2. **Select your audio file** (MP3, WAV, etc.)
3. **Select your script file** (TXT file with your dialogue)
4. **Choose where to save** the output
5. **Pick your language** (English, Spanish, etc.)
6. **Choose video format** (16:9 for YouTube, 9:16 for TikTok, etc.)

### Generate Videos
1. **Click "Batch Process All"**
2. **Wait for processing** (can take a few minutes)
3. **Find your videos** in the output folder you chose!

## 📁 What Files You Need

### Audio File
- **Formats**: MP3, WAV, M4A, FLAC
- **Quality**: Clear audio = better results

### Script File (TXT)
- **Create a text file** with your dialogue
- **Example**:
  ```
  Hello, welcome to our presentation.
  Today we will discuss important topics.
  Thank you for your attention.
  ```

## 🎬 Video Formats Available

- **16:9 (1920x1080)**: Perfect for YouTube, desktop viewing
- **9:16 (1080x1920)**: Perfect for TikTok, Instagram Stories, mobile
- **4:5 (1080x1350)**: Perfect for Instagram posts

## ⚡ Tips for Best Results

- **Clear audio** = better transcription
- **Start with "medium" model** (good balance of speed/accuracy)
- **Use "tiny" model** if processing is slow
- **Use "large" model** for best accuracy (but slower)

## 🆘 Common Problems & Solutions

### "Python not found"
- **Solution**: Reinstall Python and check "Add Python to PATH"

### "FFmpeg not found"
- **Solution**: Make sure you extracted FFmpeg to `C:\ffmpeg`

### "ImageMagick error"
- **Solution**: Copy `magick.exe` to `C:\ImageMagick\`

### "App is slow"
- **Solution**: Use smaller model (tiny/base) or close other programs

### "No audio in video"
- **Solution**: Make sure your audio file is not corrupted

## 🎉 You're Ready!

1. **Follow the installation steps above**
2. **Run `python launcher.py`**
3. **Add your audio and script files**
4. **Click "Batch Process All"**
5. **Enjoy your videos with subtitles!**

## 📞 Need Help?

- **Check this guide again**
- **Ask me directly** (I made this app!)
- **Try the test file**: `python test_app.py`

---

**Made with ❤️ by [Your Name]**

*This app uses AI to automatically generate subtitles and create videos. It's perfect for content creators, educators, and anyone who wants to add professional subtitles to their videos!* 