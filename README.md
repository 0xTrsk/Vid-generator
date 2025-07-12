# Advanced Subtitle Generator

A powerful Python application that automatically generates synchronized subtitles from audio files using AI transcription, with support for multiple video formats and batch processing.

## Features

- **AI-Powered Transcription**: Uses OpenAI Whisper and WhisperX for accurate speech-to-text conversion
- **Multiple Model Options**: Choose from tiny, base, small, medium, large, large-v2, or large-v3 Whisper models
- **Multi-Format Video Generation**: Supports 16:9 (1920x1080), 9:16 (1080x1920), and 4:5 (1080x1350) formats
- **Smart Text Wrapping**: Automatically wraps text to fit different video formats
- **Batch Processing**: Process multiple audio/script pairs simultaneously
- **Typewriter Animation**: Creates engaging word-by-word typewriter effects
- **Multi-Language Support**: Supports English, Spanish, French, Italian, Czech, Slovak, Hungarian, German, Polish, and Dutch
- **Text Correction**: Built-in correction system for common transcription errors
- **Professional Timing**: Advanced subtitle timing with lead-in/lead-out adjustments

## Installation

### Prerequisites

1. **Python 3.8+** installed on your system
2. **FFmpeg** installed and added to PATH
3. **ImageMagick** installed for text rendering

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/advanced-subtitle-generator.git
   cd advanced-subtitle-generator
   ```

2. **Install FFmpeg**:
   - **Windows**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

3. **Install ImageMagick**:
   - **Windows**: Download from [https://imagemagick.org/script/download.php](https://imagemagick.org/script/download.php)
   - **macOS**: `brew install imagemagick`
   - **Linux**: `sudo apt install imagemagick`

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Optional: Install WhisperX for enhanced timestamping**:
   ```bash
   pip install git+https://github.com/m-bain/whisperX.git
   ```

## Usage

### Quick Start

1. **Run the application**:
   ```bash
   python advanced_subtitle_generator.py
   ```

2. **Add audio/script pairs**:
   - Click "Add Pair" to select audio file, script file, and output location
   - Choose language and video format
   - Repeat for multiple files

3. **Select model size**:
   - Choose from the dropdown: tiny (39MB) to large-v3 (1550MB)
   - Larger models = better accuracy but slower processing

4. **Process**:
   - Click "Batch Process All" to generate all videos
   - Or use individual processing for single files

### File Formats

- **Audio**: MP3, WAV, M4A, FLAC
- **Scripts**: TXT files (plain text)
- **Output**: SRT subtitle files + MP4 videos with embedded subtitles

### Video Formats

- **16:9 (1920x1080)**: Standard landscape format
- **9:16 (1080x1920)**: Vertical format for mobile/social media
- **4:5 (1080x1350)**: Square format for Instagram

## Configuration

### Model Selection
- **tiny**: Fastest, least accurate (39MB)
- **base**: Good balance of speed/accuracy (74MB)
- **small**: Better accuracy (244MB)
- **medium**: Recommended default (769MB)
- **large**: High accuracy (1550MB)
- **large-v2**: Enhanced large model (1550MB)
- **large-v3**: Latest large model (1550MB)

### Text Correction
Edit the `CORRECTIONS` and `PHRASE_CORRECTIONS` dictionaries in the code to fix common transcription errors:

```python
CORRECTIONS = {
    "Doctor Mitja": "Dr. Mitja",
    "machtig": "Machtig",
    # Add your corrections here
}

PHRASE_CORRECTIONS = {
    "doctor mitja machtig": "Dr. Mitja Machtig",
    # Add phrase-level corrections here
}
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   - Ensure FFmpeg is installed and in your system PATH
   - Update the paths in the code if needed

2. **ImageMagick errors**:
   - Install ImageMagick and ensure `magick.exe` is available
   - Copy `magick.exe` to `C:\ImageMagick\` on Windows

3. **WhisperX not available**:
   - Install with: `pip install git+https://github.com/m-bain/whisperX.git`
   - The app will fall back to regular Whisper

4. **Memory issues with large models**:
   - Use smaller models (tiny, base, small) if you have limited RAM
   - Close other applications during processing

### Performance Tips

- Use GPU acceleration if available (CUDA)
- Process files in smaller batches for large datasets
- Use smaller models for faster processing
- Enable "Subtitles only" mode if you don't need video generation

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Whisper for speech recognition
- WhisperX for enhanced timestamping
- MoviePy for video generation
- NLTK for text processing

## Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Search existing issues on GitHub
3. Create a new issue with detailed information about your problem

---

**Made with ❤️ for content creators** 