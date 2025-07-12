# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
**Windows:**
- Double-click `install.bat`
- Or run: `pip install -r requirements.txt`

**Linux/Mac:**
- Run: `./install.sh`
- Or run: `pip3 install -r requirements.txt`

### 2. Run the Application
**Option A - Use Launcher (Recommended):**
```bash
python launcher.py
```

**Option B - Direct Launch:**
```bash
# Basic version (fast, simple timing)
python subtitle_generator.py

# Advanced version (speech recognition, better timing)
python advanced_subtitle_generator.py
```

### 3. Generate Subtitles
1. **Select Script File**: Choose a `.txt` file with your dialogue
2. **Select Audio File**: Choose your audio file (MP3, WAV, etc.)
3. **Choose Output**: Select where to save the `.srt` file
4. **Click Generate**: Wait for processing to complete

## 📁 File Structure
```
Sub program/
├── launcher.py                    # Main launcher (start here!)
├── subtitle_generator.py          # Basic version
├── advanced_subtitle_generator.py # Advanced version with speech recognition
├── requirements.txt               # Python dependencies
├── install.bat                    # Windows installer
├── install.sh                     # Linux/Mac installer
├── sample_script.txt              # Example script file
├── test_app.py                    # Test script
├── README.md                      # Full documentation
└── QUICK_START.md                 # This file
```

## 🎯 Sample Usage

1. **Create a script file** (e.g., `my_script.txt`):
   ```
   Hello, welcome to our presentation.
   Today we will discuss important topics.
   Thank you for your attention.
   ```

2. **Have an audio file** (e.g., `presentation.mp3`)

3. **Run the app** and select both files

4. **Get subtitles** in `.srt` format!

## ⚡ Tips
- **Basic Version**: Fast, works offline, good for most cases
- **Advanced Version**: Better timing, requires internet for speech recognition
- **Audio Quality**: Clear audio = better results
- **Script Format**: One line = one subtitle

## 🆘 Need Help?
- Check `README.md` for detailed documentation
- Run `python test_app.py` to diagnose issues
- Ensure Python 3.7+ is installed

## 🎉 You're Ready!
Start with `python launcher.py` and enjoy creating subtitles! 