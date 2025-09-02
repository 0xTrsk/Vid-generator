import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pydub import AudioSegment
# Explicitly set the path to FFmpeg if it's not in the system's PATH
# Check for bundled ffmpeg folder first, then local, then system PATH
import os
import sys

def get_bundled_path():
    """Get the path to bundled resources (for PyInstaller)"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))

base_path = get_bundled_path()

# Try bundled FFmpeg first (for PyInstaller)
bundled_ffmpeg = os.path.join(base_path, "ffmpeg", "ffmpeg.exe")
bundled_ffprobe = os.path.join(base_path, "ffmpeg", "ffprobe.exe")

# Try local ffmpeg folder second
local_ffmpeg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "ffmpeg.exe")
local_ffprobe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "ffprobe.exe")

# Set FFmpeg paths
if os.path.exists(bundled_ffmpeg) and os.path.exists(bundled_ffprobe):
    # Use bundled portable FFmpeg
    AudioSegment.converter = bundled_ffmpeg
    AudioSegment.ffprobe = bundled_ffprobe
    print(f"Using bundled FFmpeg: {bundled_ffmpeg}")
elif os.path.exists(local_ffmpeg) and os.path.exists(local_ffprobe):
    # Use local portable FFmpeg
    AudioSegment.converter = local_ffmpeg
    AudioSegment.ffprobe = local_ffprobe
    print(f"Using local FFmpeg: {local_ffmpeg}")
else:
    # Fall back to system FFmpeg
    AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
    AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"
    print("Using system FFmpeg")
import pysrt
import threading
import json
from datetime import datetime, timedelta
import re
import difflib
import whisper
from moviepy.editor import TextClip, CompositeVideoClip, concatenate_videoclips, ColorClip, AudioFileClip
import moviepy.config as mpy_config

# Try bundled ImageMagick first, then local, then system
bundled_magick = os.path.join(base_path, "imagemagick", "magick.exe")
local_magick = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagemagick", "magick.exe")
system_magick = r"C:\ImageMagick\magick.exe"

# Function to test if ImageMagick works
def test_imagemagick(path):
    """Test if ImageMagick at the given path works correctly"""
    try:
        import subprocess
        result = subprocess.run([path, '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

# Set ImageMagick path with testing
magick_path = None
if os.path.exists(bundled_magick) and test_imagemagick(bundled_magick):
    magick_path = bundled_magick
    print(f"Using bundled ImageMagick: {bundled_magick}")
elif os.path.exists(local_magick) and test_imagemagick(local_magick):
    magick_path = local_magick
    print(f"Using local ImageMagick: {local_magick}")
elif os.path.exists(system_magick) and test_imagemagick(system_magick):
    magick_path = system_magick
    print(f"Using system ImageMagick: {system_magick}")
else:
    # Try to find ImageMagick in PATH as last resort
    try:
        import subprocess
        result = subprocess.run(['magick', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Get the full path
            result = subprocess.run(['where', 'magick'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                path_candidate = result.stdout.strip().split('\n')[0]
                if test_imagemagick(path_candidate):
                    magick_path = path_candidate
                    print(f"Using ImageMagick from PATH: {magick_path}")
    except:
        pass

if magick_path:
    mpy_config.change_settings({"IMAGEMAGICK_BINARY": magick_path})
    os.environ["IMAGEMAGICK_BINARY"] = magick_path
    print(f"✓ ImageMagick configured successfully")
else:
    print("⚠ Warning: No working ImageMagick found. Video generation may fail.")
    print("   Please ensure ImageMagick is installed and accessible.")

# Try to import WhisperX for better timestamping
try:
    import whisperx
    WHISPERX_AVAILABLE = True
    print("✓ WhisperX is available for enhanced timestamping")
except ImportError:
    WHISPERX_AVAILABLE = False
    print("⚠ WhisperX not available, using regular Whisper")
    print("  Install with: pip install git+https://github.com/m-bain/whisperX.git")

# Configure MoviePy to find ImageMagick
# Note: The bundled/local/system path is already configured above
# This ensures we use the correct ImageMagick installation

# Example correction dictionary
# KEY (wrong text) : VALUE (correct text)
CORRECTIONS = {
    "Doctor Mitja": "Dr. Mitja",
    "machtig": "Machtig",
    "MOVE-UK-SCR30-2": "Move UK"  # Example of correcting other jargon
}

# Example phrase-level corrections (lowercase keys)
PHRASE_CORRECTIONS = {
    "doctor mitja machtig": "Dr. Mitja Machtig",
    "move uk scr thirty": "Move UK SCR30"
}

class AdvancedSubtitleGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Subtitle Generator")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')
        
        # Supported languages
        self.LANGUAGES = {
            "English": "en", "Spanish": "es", "French": "fr",
            "Italian": "it", "Czech": "cs", "Slovak": "sk",
            "Hungarian": "hu", "German": "de", "Polish": "pl",
            "Dutch": "nl",
        }
        self.NLTK_LANGUAGES = {
            "English": "english", "Spanish": "spanish", "French": "french",
            "Italian": "italian", "Czech": "czech", "German": "german",
            "Polish": "polish", "Dutch": "dutch",
        }
        
        # Download NLTK data if necessary
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
        except (ImportError, LookupError):
            try:
                import nltk # Re-import in case of ImportError
                messagebox.showinfo("NLTK Data Download", "The application needs to download language data for sentence splitting. This may take a moment.")
                nltk.download('punkt', quiet=True) # Download quietly
            except ImportError:
                messagebox.showwarning("NLTK Missing", "The 'nltk' library is not installed. Sentence splitting will be less accurate. Please run: pip install nltk")
            except Exception as e:
                messagebox.showwarning("Download Failed", f"Could not download NLTK data: {e}. Sentence splitting will be less accurate.")

        # Variables
        self.script_file = tk.StringVar()
        self.audio_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.language = tk.StringVar(value="English")
        self.model_size = tk.StringVar(value="medium")  # Add model size selection
        self.audio_files = []  # For batch processing
        self.audio_script_pairs = []  # List of dicts: {"audio": ..., "script": ..., "output": ..., "language": ...}
        self.timing_offset = tk.DoubleVar(value=0.0)  # Restore timing offset for batch mode
        
        # Initialize Whisper model (OpenAI Whisper)
        self.whisper_model = None
        
        # Subtitles only checkbox
        self.subtitles_only = tk.BooleanVar(value=False)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Restore original simple gray UI
        self.root.configure(bg='#f0f0f0')

        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Advanced Subtitle Generator", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Audio/script pair selection
        ttk.Label(main_frame, text="Audio/Script/Output/Language Pairs:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.pair_listbox = tk.Listbox(main_frame, height=7, width=80)
        self.pair_listbox.grid(row=1, column=1, sticky="ew", padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="Add Pair", command=self.add_audio_script_pair).grid(row=1, column=2, pady=5)

        # Batch process button
        batch_btn = ttk.Button(main_frame, text="Batch Process All", command=self.batch_process_videos)
        batch_btn.grid(row=2, column=0, columnspan=3, pady=10)

        # Progress bar
        ttk.Label(main_frame, text="Progress:").grid(row=3, column=0, sticky=tk.W, pady=(20, 5))
        progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        progress_bar.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=(20, 5))

        # Status label
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=4, column=0, columnspan=3, pady=5)

        # Help text
        help_text = "Add each audio/script/output/language pair you want to process, then click 'Batch Process All'."
        help_label = ttk.Label(main_frame, text=help_text, wraplength=800, foreground='gray')
        help_label.grid(row=5, column=0, columnspan=3, pady=(0, 20))

        # Preview area
        preview_frame = ttk.LabelFrame(main_frame, text="Generated Subtitles Preview", padding=10)
        preview_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(20, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=15, width=80)
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        main_frame.rowconfigure(6, weight=1)
        
        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="Model Settings", padding=10)
        model_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        ttk.Label(model_frame, text="Whisper Model Size:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_size, 
                                  values=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], 
                                  state="readonly", width=15)
        model_combo.grid(row=0, column=1, sticky=tk.W)
        model_combo.set("medium")
        
        # Animation style selection
        ttk.Label(model_frame, text="Animation Style:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.animation_style = tk.StringVar(value="typewriter")
        animation_combo = ttk.Combobox(model_frame, textvariable=self.animation_style,
                                      values=list(get_animation_style_options().keys()),
                                      state="readonly", width=20)
        animation_combo.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Lead-in timing control
        ttk.Label(model_frame, text="Word Lead-in (ms):").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.word_lead_in = tk.DoubleVar(value=100)  # 100ms default
        lead_in_spinbox = ttk.Spinbox(model_frame, from_=0, to=500, increment=25, 
                                      textvariable=self.word_lead_in, width=10)
        lead_in_spinbox.grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        
        # Max words per subtitle control
        ttk.Label(model_frame, text="Max Words/Subtitle:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.max_words_per_subtitle = tk.IntVar(value=8)  # 8 words default
        max_words_spinbox = ttk.Spinbox(model_frame, from_=6, to=12, increment=1, 
                                        textvariable=self.max_words_per_subtitle, width=10)
        max_words_spinbox.grid(row=3, column=1, sticky=tk.W, pady=(10, 0))
        
        # Model info
        model_info = "tiny (39MB) < base (74MB) < small (244MB) < medium (769MB) < large (1550MB) < large-v2 (1550MB) < large-v3 (1550MB)"
        ttk.Label(model_frame, text=model_info, foreground='gray', wraplength=600).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Animation style info
        animation_info = "Choose from: typewriter, fade_in, slide_up, bounce, glitch, wave, zoom_in, typewriter_enhanced"
        ttk.Label(model_frame, text=animation_info, foreground='gray', wraplength=600).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Detailed animation help
        animation_help = """Animation Styles:
• typewriter: Classic word-by-word appearance
• fade_in: Smooth fade in effect
• slide_up: Words slide up from bottom
• bounce: Bouncy elastic entrance
• glitch: Digital glitch effect
• wave: Wave pattern appearance
• zoom_in: Zoom in effect
• typewriter_enhanced: Typewriter with blinking cursor"""
        
        help_label = ttk.Label(model_frame, text=animation_help, foreground='blue', wraplength=600, justify='left')
        help_label.grid(row=6, column=0, columnspan=2, sticky="w", pady=(5, 0))
        
        # Timing help
        timing_help = "Word Lead-in: How many milliseconds before each word is spoken that it appears on screen (0-500ms)"
        timing_label = ttk.Label(model_frame, text=timing_help, foreground='green', wraplength=600, justify='left')
        timing_label.grid(row=7, column=0, columnspan=2, sticky="w", pady=(5, 0))
        
        # Subtitle grouping help
        grouping_help = "Max Words/Subtitle: Controls how many words appear per subtitle line (6-12, default 8 for 2-line display)"
        grouping_label = ttk.Label(model_frame, text=grouping_help, foreground='purple', wraplength=600, justify='left')
        grouping_label.grid(row=8, column=0, columnspan=2, sticky="w", pady=(5, 0))
        
        # Subtitles only checkbox
        subtitles_only_cb = ttk.Checkbutton(main_frame, text="Subtitles only (do not create video)", variable=self.subtitles_only)
        subtitles_only_cb.grid(row=9, column=0, columnspan=3, pady=(5, 0), sticky=tk.W)
        
    def browse_script(self):
        filename = filedialog.askopenfilename(
            title="Select Script File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.script_file.set(filename)
            
    def browse_audio(self):
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.flac"), ("All files", "*.*")]
        )
        if filename:
            self.audio_file.set(filename)
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Subtitles As",
            defaultextension=".srt",
            filetypes=[("SubRip files", "*.srt"), ("All files", "*.*")]
        )
        if filename:
            self.output_file.set(filename)
            
    def browse_audio_batch(self):
        filenames = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.flac"), ("All files", "*.*")]
        )
        if filenames:
            self.audio_files = list(filenames)
            self.audio_file.set("; ".join(self.audio_files))  # For display

    def generate_subtitles(self):
        if not self.script_file.get() or not self.audio_file.get() or not self.output_file.get():
            messagebox.showerror("Error", "Please select all required files.")
            return
            
        # Start generation in a separate thread
        thread = threading.Thread(target=self._generate_subtitles_thread)
        thread.daemon = True
        thread.start()
        
    def _generate_subtitles_thread(self):
        try:
            self.status_var.set("Loading script...")
            self.progress_var.set(10)

            # Load and process script (for preview only)
            script_lines = self.load_script()

            self.status_var.set("Loading Whisper model...")
            self.progress_var.set(20)

            # Load Whisper model
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model(self.model_size.get())

            # Transcribe audio with Whisper or WhisperX
            language_code = self.LANGUAGES.get(self.language.get(), "en")

            if WHISPERX_AVAILABLE:
                self.status_var.set("Transcribing audio with WhisperX (better timestamping)...")
                self.progress_var.set(40)
                segments = self._transcribe_with_whisperx(language_code, self.model_size.get())
            else:
                self.status_var.set("Transcribing audio with Whisper...")
                self.progress_var.set(40)
                result = self.whisper_model.transcribe(
                    self.audio_file.get(),
                    language=language_code,
                    word_timestamps=True
                )
                segments = result["segments"] if "segments" in result else []

            self.status_var.set("Aligning transcription and grouping by punctuation...")
            self.progress_var.set(70)

            # --- Apply phrase-level corrections here ---
            word_segments = force_correct_phrases(segments, PHRASE_CORRECTIONS)

            # Store word segments for video creation
            self.current_word_segments = word_segments.copy()

            # 1. Align segments if WhisperX is used
            if WHISPERX_AVAILABLE:
                audio_file_path = self.audio_file.get()
                try:
                    lang = language_code
                    if hasattr(segments, 'language'):
                        lang = segments.language
                    import whisperx
                    model_a, metadata = whisperx.load_align_model(language_code=lang, device="cpu")
                    aligned_result = whisperx.align(segments, model_a, metadata, audio_file_path, "cpu", return_char_alignments=False)
                    word_segments = aligned_result.get('word_segments', [])
                    # Update stored word segments with aligned data
                    self.current_word_segments = word_segments.copy()
                except Exception as e:
                    print(f"WhisperX alignment failed: {e}")
                    word_segments = []
            else:
                word_segments = []
                for segment in segments:
                    if 'words' in segment:
                        for w in segment['words']:
                            if w['word'].strip():
                                word_segments.append(w)
                    elif hasattr(segment, 'words') and segment.words:
                        for w in segment.words:
                            if w.word.strip():
                                word_segments.append({'word': w.word, 'start': w.start, 'end': w.end})
                    elif 'text' in segment:
                        text = segment['text']
                        start_time = segment['start']
                        end_time = segment['end']
                        word_list = text.split()
                        if word_list:
                            duration_per_word = (end_time - start_time) / len(word_list)
                            for i, word in enumerate(word_list):
                                word_start = start_time + (i * duration_per_word)
                                word_end = word_start + duration_per_word
                                word_segments.append({'word': word, 'start': word_start, 'end': word_end})

            # --- Apply phrase-level corrections here ---
            word_segments = force_correct_phrases(word_segments, PHRASE_CORRECTIONS)

            # 2. Group words by punctuation
            logical_subtitles = self.group_words_by_punctuation(word_segments, max_words=self.max_words_per_subtitle.get())

            # 2.5. ✍️ NEW: Correct known names and errors
            logical_subtitles = correct_known_errors(logical_subtitles, CORRECTIONS)

            # 3. Adjust overlaps
            logical_subtitles = adjust_subtitle_overlaps(logical_subtitles, min_gap_seconds=0.1)

            # 4. ⏱️ NEW: Add padding for better perceived sync
            logical_subtitles = add_subtitle_padding(logical_subtitles)

            # 4.5. Apply lead-in and lead-out for better readability
            audio_duration = self.get_audio_duration()  # Clamp subtitles to audio length
            logical_subtitles = apply_subtitle_lead_in_out(logical_subtitles, lead_in=0.4, lead_out=0.4, audio_duration=audio_duration)

            # 4.6. Filter out subtitles that are empty or only punctuation
            import string
            def is_meaningful_sub(sub):
                text = sub['text'].strip()
                return text and any(c.isalnum() for c in text)
            logical_subtitles = [sub for sub in logical_subtitles if is_meaningful_sub(sub)]

            # 4.7. Merge one-word subtitles with the previous subtitle
            def merge_one_word_subs(subs):
                if not subs:
                    return subs
                merged = [subs[0]]
                for sub in subs[1:]:
                    if len(sub['text'].strip().split()) == 1 and len(merged[-1]['text'].strip().split()) > 0:
                        # Merge with previous
                        merged[-1]['text'] = merged[-1]['text'].rstrip() + ' ' + sub['text'].lstrip()
                        merged[-1]['end'] = sub['end']  # Extend end time
                    else:
                        merged.append(sub)
                return merged
            logical_subtitles = merge_one_word_subs(logical_subtitles)

            # 5. Convert to pysrt.SubRipItem objects
            subtitles = []
            n = len(logical_subtitles)
            for idx, sub in enumerate(logical_subtitles, 1):
                offset = self.timing_offset.get()
                adjusted_start = max(0, sub['start'] + offset)
                # Calculate the default end time
                adjusted_end = max(adjusted_start + 0.5, sub['end'] + offset)
                # Clamp end time to the start of the next subtitle, if any
                if idx < n:
                    next_start = max(0, logical_subtitles[idx]['start'] + offset)
                    if adjusted_end > next_start:
                        adjusted_end = next_start - 0.01  # Small gap
                # Clamp end time to audio duration
                if adjusted_end > audio_duration:
                    adjusted_end = audio_duration
                if adjusted_start > audio_duration:
                    adjusted_start = audio_duration - 0.1  # Prevent start after audio ends
                if adjusted_end <= adjusted_start:
                    adjusted_end = min(audio_duration, adjusted_start + 1.5)
                subtitles.append(
                    pysrt.SubRipItem(
                        index=idx,
                        text=sub['text'],
                        start=self.seconds_to_time(adjusted_start),
                        end=self.seconds_to_time(adjusted_end)
                    )
                )

            self.status_var.set("Fixing subtitle overlaps...")
            self.progress_var.set(85)
            self.fix_subtitle_overlaps(subtitles)

            self.status_var.set("Saving subtitles...")
            self.progress_var.set(90)
            self.save_subtitles(subtitles)
            # Only create video if not subtitles only
            if not self.subtitles_only.get():
                self.status_var.set("Creating video with subtitles...")
                self.progress_var.set(95)
                self.create_video_with_subtitles(subtitles, show_success_message=True)
            self.progress_var.set(100)
            self.status_var.set("Subtitles generated successfully!" if self.subtitles_only.get() else "Subtitles and video generated successfully!")
            self.update_preview(subtitles)

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
    def load_script(self):
        """Load, clean, and parse the script into appropriately sized subtitle lines."""
        with open(self.script_file.get(), 'r', encoding='utf-8') as f:
            # Correctly replace newlines with spaces to form continuous text
            content = f.read().replace('\\n', ' ').replace('\\r', ' ').replace('\n', ' ')

        selected_lang_name = self.language.get()
        nltk_lang = self.NLTK_LANGUAGES.get(selected_lang_name)

        try:
            if not nltk_lang: raise ImportError
            import nltk.data
            tokenizer = nltk.data.load(f'tokenizers/punkt/{nltk_lang}.pickle')
            sentences = tokenizer.tokenize(content)
        except (ImportError, LookupError):
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', content) if s.strip()]

        # Post-processing to create well-formed subtitle lines
        # 1. Merge unnaturally short sentences (often fragments) with the next one.
        processed_sentences = []
        i = 0
        while i < len(sentences):
            current_sentence = sentences[i].strip()
            if not current_sentence:
                i += 1
                continue
            
            # If a sentence is very short and not the last one, merge it with the next.
            if len(current_sentence.split()) <= 2 and i + 1 < len(sentences):
                merged = current_sentence + " " + sentences[i+1].strip()
                processed_sentences.append(merged)
                i += 2  # Skip next sentence since it's merged
            else:
                processed_sentences.append(current_sentence)
                i += 1
        
        # 2. Split any remaining long sentences into lines of max_words.
        final_lines = []
        max_words = 9
        for sentence in processed_sentences:
            words = sentence.split()
            if not words: continue

            if len(words) <= max_words:
                final_lines.append(sentence)
            else:
                # Smart chunking for long sentences with exception for end-of-sentence words
                chunks = []
                i = 0
                while i < len(words):
                    # Check if we're near the end of the sentence
                    remaining_words = len(words) - i
                    
                    # Normal chunking logic
                    chunk_size = min(max_words, remaining_words)
                    
                    # Check if this chunk would end with sentence-ending punctuation
                    end_idx = i + chunk_size
                    if end_idx <= len(words):
                        chunk_words = words[i:end_idx]
                        last_chunk_word = chunk_words[-1]
                        
                        # If the chunk ends with sentence punctuation and we have more words after it,
                        # check if we should include those words to avoid single-word subtitles
                        if last_chunk_word.endswith(('.', '?', '!')):
                            # If we have 1-2 more words after this chunk, include them
                            if remaining_words > chunk_size and remaining_words - chunk_size <= 2:
                                # Extend the chunk to include the remaining words
                                chunk_size = remaining_words
                    
                    chunk = " ".join(words[i:i + chunk_size])
                    chunks.append(chunk)
                    i += chunk_size
                
                final_lines.extend(chunks)
        return [line for line in final_lines if line]
    
    def _transcribe_with_whisperx(self, language_code, model_size="medium"):
        """Transcribe audio using WhisperX with new VAD method for better timestamping."""
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load the main Whisper model
            print("Loading Whisper model...")
            model = whisperx.load_model(model_size, device, compute_type="float32")
            
            # Read the script file to use as initial_prompt for better accuracy
            prompt_text = ""
            if self.script_file.get():
                try:
                    with open(self.script_file.get(), "r", encoding="utf-8") as f:
                        prompt_text = f.read()
                    print("✓ Script loaded for improved transcription accuracy")
                except Exception as e:
                    print(f"⚠ Could not load script for initial_prompt: {e}")
            
            # Transcribe using the new method with vad_options and initial_prompt
            print("Transcribing with built-in VAD and script context...")
            result = model.transcribe(
                self.audio_file.get(),
                language=language_code,
                verbose=False
            )
            
            # Align with wav2vec2 for even better timestamps
            try:
                model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    self.audio_file.get(), 
                    device,
                    return_char_alignments=False
                )
                print("✓ WhisperX alignment completed successfully")
            except Exception as align_error:
                print(f"⚠ WhisperX alignment failed, using basic transcription: {align_error}")
            
            # Convert WhisperX segments to word-level format
            segments_with_words = []
            for segment in result["segments"]:
                if "words" in segment:
                    # WhisperX already provides word-level timestamps
                    segments_with_words.append(segment)
                else:
                    # Convert segment to word-level format
                    words = []
                    if "text" in segment:
                        # Simple word splitting for segments without word timestamps
                        word_list = segment["text"].split()
                        start_time = segment["start"]
                        end_time = segment["end"]
                        duration_per_word = (end_time - start_time) / max(1, len(word_list))
                        
                        for i, word in enumerate(word_list):
                            word_start = start_time + (i * duration_per_word)
                            word_end = word_start + duration_per_word
                            words.append({
                                "word": word,
                                "start": word_start,
                                "end": word_end
                            })
                    
                    segment["words"] = words
                    segments_with_words.append(segment)
            
            return segments_with_words
            
        except Exception as e:
            print(f"Error with WhisperX: {e}")
            print("Falling back to regular Whisper...")
            # Fallback to regular Whisper
            result = self.whisper_model.transcribe(
                self.audio_file.get(),
                language=language_code,
                word_timestamps=True
            )
            return result["segments"] if "segments" in result else []
        
    def group_words_by_punctuation(self, word_segments, max_words=8, merge_short_lines=True):
        """
        Groups word segments into subtitle lines with improved logic for better readability.
        
        Requirements:
        - Sentences end at . ? !
        - Max 8-10 words per subtitle (fits on 2 lines)
        - No 1-2 word subtitles except complete sentences
        - Smart merging of short fragments
        """
        subtitle_lines = []
        current_line_words = []
        
        # Track if we're in a sentence that's too short
        in_short_sentence = False
        sentence_start_idx = 0

        for i, word_data in enumerate(word_segments):
            word = word_data['word'].strip()
            current_line_words.append(word_data)

            # Check for sentence-ending punctuation
            if word.endswith(('.', '?', '!')):
                # We have a complete sentence
                start_time = current_line_words[0]['start']
                end_time = current_line_words[-1]['end']
                text = " ".join([w['word'] for w in current_line_words])
                
                # Always add complete sentences, even if short
                subtitle_lines.append({"text": text, "start": start_time, "end": end_time})
                current_line_words = []
                in_short_sentence = False
                
            # Check for commas - be more aggressive about merging
            elif word.endswith(','):
                # Only break on comma if we have enough words and it's not a very short fragment
                if len(current_line_words) >= 6:  # Allow longer lines before comma breaks
                    start_time = current_line_words[0]['start']
                    end_time = current_line_words[-1]['end']
                    text = " ".join([w['word'] for w in current_line_words])
                    subtitle_lines.append({"text": text, "start": start_time, "end": end_time})
                    current_line_words = []
                    in_short_sentence = False
                    
            # Check if we're exceeding the word limit
            elif len(current_line_words) >= max_words:
                # Look ahead to see if we're near the end of a sentence
                look_ahead_words = 0
                for j in range(i + 1, min(i + 4, len(word_segments))):  # Look up to 3 words ahead
                    if j < len(word_segments):
                        next_word = word_segments[j]['word'].strip()
                        if next_word.endswith(('.', '?', '!')):
                            look_ahead_words = j - i
                            break
                
                # If we're close to sentence end, wait for it
                if look_ahead_words <= 2 and look_ahead_words > 0:
                    pass  # Keep building to include the sentence ending
                else:
                    # Break here to avoid too long lines
                    start_time = current_line_words[0]['start']
                    end_time = current_line_words[-1]['end']
                    text = " ".join([w['word'] for w in current_line_words])
                    subtitle_lines.append({"text": text, "start": start_time, "end": end_time})
                    current_line_words = []
                    in_short_sentence = False

        # Handle any remaining words
        if current_line_words:
            # If we have a very short remaining fragment, try to merge with the last subtitle
            if len(current_line_words) <= 3 and subtitle_lines:
                last_subtitle = subtitle_lines[-1]
                # Merge if the combined length is reasonable
                combined_words = len(last_subtitle['text'].split()) + len(current_line_words)
                if combined_words <= max_words + 2:  # Allow slight overflow for merging
                    # Extend the last subtitle
                    last_subtitle['text'] = last_subtitle['text'] + ' ' + " ".join([w['word'] for w in current_line_words])
                    last_subtitle['end'] = current_line_words[-1]['end']
                else:
                    # Create a new subtitle for the remaining words
                    start_time = current_line_words[0]['start']
                    end_time = current_line_words[-1]['end']
                    text = " ".join([w['word'] for w in current_line_words])
                    subtitle_lines.append({"text": text, "start": start_time, "end": end_time})
            else:
                # Normal case: create subtitle for remaining words
                start_time = current_line_words[0]['start']
                end_time = current_line_words[-1]['end']
                text = " ".join([w['word'] for w in current_line_words])
                subtitle_lines.append({"text": text, "start": start_time, "end": end_time})

        # Post-process: merge very short subtitles with previous ones when possible
        final_subtitles = []
        for subtitle in subtitle_lines:
            word_count = len(subtitle['text'].split())
            
            # If this subtitle is very short (1-2 words) and not a complete sentence
            if word_count <= 2 and not subtitle['text'].strip().endswith(('.', '?', '!')):
                # Try to merge with previous subtitle if it won't make it too long
                if final_subtitles:
                    prev_subtitle = final_subtitles[-1]
                    combined_words = len(prev_subtitle['text'].split()) + word_count
                    if combined_words <= max_words + 1:  # Allow slight overflow
                        # Merge with previous subtitle
                        prev_subtitle['text'] = prev_subtitle['text'] + ' ' + subtitle['text']
                        prev_subtitle['end'] = subtitle['end']
                        continue  # Skip adding this as a separate subtitle
            
            # Add the subtitle as-is
            final_subtitles.append(subtitle)

        return final_subtitles
    
    def fix_subtitle_overlaps(self, subtitles):
        """Fix any overlapping subtitles and ensure proper spacing."""
        if len(subtitles) < 2:
            return
            
        # Sort subtitles by start time to ensure proper order
        subtitles.sort(key=lambda x: x.start.ordinal)
        
        # Ensure minimum spacing between subtitles and fix overlaps
        min_gap = 500  # 500ms minimum gap between subtitles (increased from 200ms)
        
        for i in range(len(subtitles) - 1):
            current_end = subtitles[i].end.ordinal
            next_start = subtitles[i+1].start.ordinal
            
            # If there's an overlap or insufficient gap
            if next_start <= current_end + min_gap:
                # Calculate new timing to avoid overlap
                new_next_start = current_end + min_gap
                subtitles[i+1].start.ordinal = new_next_start
                
                # Ensure the next subtitle still has reasonable duration
                min_duration = 1500  # 1.5 seconds minimum
                if subtitles[i+1].end.ordinal <= new_next_start + min_duration:
                    subtitles[i+1].end.ordinal = new_next_start + min_duration
        
        # Ensure all subtitles have reasonable duration
        for subtitle in subtitles:
            start_ms = subtitle.start.ordinal
            end_ms = subtitle.end.ordinal
            min_duration = 1500  # 1.5 seconds minimum
            
            if end_ms <= start_ms:
                subtitle.end.ordinal = start_ms + min_duration
            elif end_ms - start_ms < min_duration:
                subtitle.end.ordinal = start_ms + min_duration
        
        # Reassign indices in time order
        for idx, subtitle in enumerate(subtitles, 1):
            subtitle.index = idx
    
    def get_audio_duration(self):
        """Get the duration of the audio file in seconds"""
        try:
            audio = AudioSegment.from_file(self.audio_file.get())
            return len(audio) / 1000.0  # Convert to seconds
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return 60.0  # Default fallback duration
        
    def seconds_to_time(self, seconds):
        """Convert seconds to pysrt time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return pysrt.SubRipTime(hours, minutes, secs, millisecs)
        
    def save_subtitles(self, subtitles):
        """Save subtitles to file"""
        subs = pysrt.SubRipFile(items=subtitles)
        subs.save(self.output_file.get(), encoding='utf-8')
        
    def update_preview(self, subtitles):
        """Update the preview text area"""
        self.preview_text.delete(1.0, tk.END)
        
        preview_text = ""
        for subtitle in subtitles:
            start_str = f"{subtitle.start.hours:02d}:{subtitle.start.minutes:02d}:{subtitle.start.seconds:02d},{subtitle.start.milliseconds:03d}"
            end_str = f"{subtitle.end.hours:02d}:{subtitle.end.minutes:02d}:{subtitle.end.seconds:02d},{subtitle.end.milliseconds:03d}"
            
            preview_text += f"{subtitle.index}\n"
            preview_text += f"{start_str} --> {end_str}\n"
            preview_text += f"{subtitle.text}\n\n"
            
        self.preview_text.insert(1.0, preview_text)

    def create_video_with_subtitles(self, subtitles, show_success_message=True, video_format="16:9 (1920x1080)", animation_style="typewriter", word_lead_in=100):
        """Create a video with optional scene and typewriter subtitles, or classic fast mode if scene is not used."""
        print("=== Starting video creation process ===")
        
        # Set the lead-in timing for this video
        self.current_word_lead_in = word_lead_in
        
        try:
            # Check if ImageMagick is available for text rendering
            print("Checking ImageMagick availability...")
            imagemagick_available = False
            
            # First check if we have a configured ImageMagick path
            if 'IMAGEMAGICK_BINARY' in os.environ:
                configured_path = os.environ['IMAGEMAGICK_BINARY']
                print(f"Testing configured ImageMagick: {configured_path}")
                try:
                    test_text = TextClip("TEST", fontsize=20, color='black')
                    test_text.close()
                    imagemagick_available = True
                    print("✓ ImageMagick is working correctly")
                except Exception as e:
                    print(f"✗ Configured ImageMagick failed: {e}")
            
            # If configured path failed, try to find a working ImageMagick
            if not imagemagick_available:
                print("Trying to find working ImageMagick...")
                possible_paths = [
                    os.path.join(base_path, "imagemagick", "magick.exe"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagemagick", "magick.exe"),
                    r"C:\ImageMagick\magick.exe",
                    r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
                    r"C:\Program Files\ImageMagick-7.1.1-Q16\magick.exe"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        print(f"Testing ImageMagick at: {path}")
                        try:
                            # Temporarily set the path
                            old_path = os.environ.get('IMAGEMAGICK_BINARY')
                            os.environ['IMAGEMAGICK_BINARY'] = path
                            mpy_config.change_settings({"IMAGEMAGICK_BINARY": path})
                            
                            test_text = TextClip("TEST", fontsize=20, color='black')
                            test_text.close()
                            imagemagick_available = True
                            print(f"✓ Found working ImageMagick at: {path}")
                            break
                        except Exception as e:
                            print(f"✗ Failed at {path}: {e}")
                            # Restore old path if it existed
                            if old_path:
                                os.environ['IMAGEMAGICK_BINARY'] = old_path
                                mpy_config.change_settings({"IMAGEMAGICK_BINARY": old_path})
                            else:
                                os.environ.pop('IMAGEMAGICK_BINARY', None)
            
            if not imagemagick_available:
                print("✗ No working ImageMagick found")
                messagebox.showwarning("ImageMagick Required", 
                    "ImageMagick is not working properly, which is required for text rendering.\n\n"
                    "Possible solutions:\n"
                    "1. Ensure ImageMagick is installed correctly\n"
                    "2. Try reinstalling ImageMagick\n"
                    "3. Check if the bundled ImageMagick is present in the program folder\n\n"
                    "For now, only the SRT subtitle file will be created.")
                return

            audio_path = self.audio_file.get()
            if not audio_path:
                messagebox.showerror("Error", "Audio file path is missing.")
                return

            output_video_path = os.path.splitext(self.output_file.get())[0] + "_video.mp4"

            # Load audio and get duration
            self.status_var.set("Loading audio file...")
            audio = AudioFileClip(audio_path)
            audio_duration = audio.duration

            # Determine video size based on format
            if video_format == "16:9 (1920x1080)":
                size = (1920, 1080)
            elif video_format == "9:16 (1080x1920)":
                size = (1080, 1920)
            elif video_format == "4:5 (1080x1350)":
                size = (1080, 1350)
            else:
                size = (1920, 1080)

            # --- Get scene and toggle ---
            scene_path = None
            use_scene = False
            for pair in getattr(self, 'audio_script_pairs', []):
                if pair.get('audio') == audio_path and pair.get('script') == self.script_file.get() and pair.get('output') == self.output_file.get():
                    scene_path = pair.get('scene')
                    use_scene = pair.get('use_scene', False)
                    break
            if not scene_path:
                scene_path = None

            # --- CLASSIC FAST MODE (no scene) ---
            if not use_scene or not scene_path:
                print("Using classic fast mode: white background, typewriter subtitles, no scene.")
                background_clip = ColorClip(size=size, color=(255, 255, 255)).set_duration(audio_duration)
                all_clips = [background_clip]
                for i, sub in enumerate(subtitles):
                    sub_data = {
                        'text': str(sub.text),
                        'start': sub.start.ordinal / 1000.0,
                        'end': sub.end.ordinal / 1000.0
                    }
                    # Find corresponding word timings for this subtitle
                    subtitle_word_timings = []
                    if hasattr(self, 'current_word_segments') and self.current_word_segments:
                        # Find words that belong to this subtitle
                        sub_start = sub_data['start']
                        sub_end = sub_data['end']
                        for word_seg in self.current_word_segments:
                            if (word_seg['start'] >= sub_start and word_seg['start'] < sub_end) or \
                               (word_seg['end'] > sub_start and word_seg['end'] <= sub_end):
                                subtitle_word_timings.append(word_seg)
                    
                    typewriter_clip = create_word_typewriter_clip(
                        sub_data, size, video_format=video_format, 
                        animation_style=animation_style, 
                        word_timings=subtitle_word_timings,
                        app_instance=self
                    )
                    if typewriter_clip:
                        all_clips.append(typewriter_clip)
                final = CompositeVideoClip(all_clips).set_audio(audio).set_duration(audio_duration)
            else:
                # --- SCENE MODE: scene at start, typewriter subtitles on both scene and white background ---
                print("Using scene mode: scene at start, typewriter subtitles on both scene and white background.")
                scene_clip = None
                scene_duration = 0
                scene_loaded = False
                try:
                    from moviepy.editor import VideoFileClip
                    scene_clip = VideoFileClip(scene_path)
                    scene_clip = scene_clip.resize(newsize=size)
                    scene_duration = scene_clip.duration
                    scene_loaded = True
                    print(f"✓ Scene loaded: {scene_path} ({scene_duration:.2f}s)")
                except Exception as e:
                    print(f"⚠ Could not load scene: {e}")
                    messagebox.showwarning("Scene Error", f"Could not load scene video. Only white background will be used.\nError: {e}")
                    scene_clip = None
                    scene_duration = 0
                    scene_loaded = False
                white_duration = max(0, audio_duration - scene_duration)
                white_clip = ColorClip(size=size, color=(255, 255, 255), duration=white_duration).set_start(scene_duration)
                # Prepare subtitle clips for both scene and white background
                subtitle_clips = []
                for i, sub in enumerate(subtitles):
                    sub_start = sub.start.ordinal / 1000.0
                    sub_end = sub.end.ordinal / 1000.0
                    # If subtitle ends before scene, skip
                    if sub_end <= 0:
                        continue
                    # If subtitle starts before scene ends and ends after scene starts, split if needed
                    if sub_start < scene_duration:
                        # Clamp to scene duration
                        scene_sub_start = max(sub_start, 0)
                        scene_sub_end = min(sub_end, scene_duration)
                        if scene_sub_end > scene_sub_start:
                            sub_data_scene = {
                                'text': str(sub.text),
                                'start': scene_sub_start,
                                'end': scene_sub_end
                            }
                            # On scene: white bg for subtitle
                            typewriter_clip_scene = create_word_typewriter_clip(
                                sub_data_scene, size, font="Arial-Bold", fontsize=70 if size[1] >= 1080 else 50, 
                                color='black', bg_color='white', video_format=video_format, 
                                animation_style=animation_style, word_timings=subtitle_word_timings,
                                app_instance=self
                            )
                            if typewriter_clip_scene:
                                subtitle_clips.append(typewriter_clip_scene.set_start(scene_sub_start))
                        # If subtitle continues onto white background
                        if sub_end > scene_duration:
                            white_sub_start = scene_duration
                            white_sub_end = sub_end
                            sub_data_white = {
                                'text': str(sub.text),
                                'start': 0,
                                'end': white_sub_end - white_sub_start
                            }
                            typewriter_clip_white = create_word_typewriter_clip(
                                sub_data_white, size, font="Arial-Bold", fontsize=70 if size[1] >= 1080 else 50, 
                                color='black', bg_color=None, video_format=video_format, 
                                animation_style=animation_style, word_timings=subtitle_word_timings,
                                app_instance=self
                            )
                            if typewriter_clip_white:
                                subtitle_clips.append(typewriter_clip_white.set_start(white_sub_start))
                    else:
                        # Only on white background
                        white_sub_start = sub_start
                        white_sub_end = sub_end
                        sub_data_white = {
                            'text': str(sub.text),
                            'start': white_sub_start - scene_duration,
                            'end': white_sub_end - scene_duration
                        }
                        typewriter_clip_white = create_word_typewriter_clip(
                            sub_data_white, size, font="Arial-Bold", fontsize=70 if size[1] >= 1080 else 50, 
                            color='black', bg_color=None, video_format=video_format, 
                            animation_style=animation_style, word_timings=subtitle_word_timings,
                            app_instance=self
                        )
                        if typewriter_clip_white:
                            subtitle_clips.append(typewriter_clip_white.set_start(white_sub_start))
                # Compose scene and white background
                clips_to_concat = []
                if scene_loaded and scene_clip:
                    scene_with_subs = CompositeVideoClip([scene_clip] + [c for c in subtitle_clips if c.start < scene_duration]).set_duration(scene_duration)
                    clips_to_concat.append(scene_with_subs)
                if white_duration > 0:
                    white_with_subs = CompositeVideoClip([white_clip] + [c for c in subtitle_clips if c.start >= scene_duration]).set_duration(white_duration)
                    clips_to_concat.append(white_with_subs)
                final = concatenate_videoclips(clips_to_concat, method="compose").set_audio(audio).set_duration(audio_duration)

            # Export
            self.status_var.set("Exporting video...")
            print(f"Starting video export to: {output_video_path}")
            print(f"Final video duration: {final.duration}")
            print(f"Final video size: {final.size}")

            final.write_videofile(
                output_video_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=30
            )

            self.status_var.set("Video created successfully!")
            if show_success_message:
                messagebox.showinfo("Success", f"Video with integrated subtitles created successfully!\nSaved to: {output_video_path}")

        except ImportError:
            messagebox.showerror("Error", "MoviePy is not installed. Please install it with: pip install moviepy")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during video creation: {str(e)}")
            print(f"Full error details: {e}")
            import traceback
            traceback.print_exc()

    def add_audio_script_pair(self):
        audio_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.flac"), ("All files", "*.*")]
        )
        if not audio_path:
            return
        script_path = filedialog.askopenfilename(
            title="Select Script File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not script_path:
            return
        # Scene selection (optional)
        scene_path = filedialog.askopenfilename(
            title="Select Scene Video (Optional)",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")]
        )
        if not scene_path:
            scene_path = None
        # Output file
        output_path = filedialog.asksaveasfilename(
            title="Save Subtitles As",
            defaultextension=".srt",
            filetypes=[("SubRip files", "*.srt"), ("All files", "*.*")]
        )
        if not output_path:
            return
        # Language and format selection dialog
        lang_window = tk.Toplevel(self.root)
        lang_window.title("Select Language and Video Format")
        tk.Label(lang_window, text="Language:").pack(padx=10, pady=5)
        lang_var = tk.StringVar(value="English")
        lang_combo = ttk.Combobox(lang_window, textvariable=lang_var, values=list(self.LANGUAGES.keys()))
        lang_combo.pack(padx=10, pady=5)

        # Video format checkboxes
        tk.Label(lang_window, text="Video Format:").pack(padx=10, pady=(10, 0))
        format_vars = {
            "16:9 (1920x1080)": tk.BooleanVar(value=True),
            "9:16 (1080x1920)": tk.BooleanVar(value=False),
            "4:5 (1080x1350)": tk.BooleanVar(value=False),
        }
        format_checks = {}
        for key, var in format_vars.items():
            cb = tk.Checkbutton(lang_window, text=key, variable=var)
            cb.pack(anchor='w', padx=20)
            format_checks[key] = cb

        # Scene toggle
        use_scene_var = tk.BooleanVar(value=bool(scene_path))
        scene_toggle = tk.Checkbutton(lang_window, text="Add scene at start (with typewriter subtitles on scene)", variable=use_scene_var)
        scene_toggle.pack(anchor='w', padx=10, pady=(10, 0))

        def get_selected_formats():
            selected = [key for key, var in format_vars.items() if var.get()]
            return selected if selected else ["16:9 (1920x1080)"]

        def set_lang():
            language = lang_var.get()
            selected_formats = get_selected_formats()
            first = True
            for video_format in selected_formats:
                base, ext = os.path.splitext(output_path)
                if "16:9" in video_format:
                    format_suffix = "_16x9"
                elif "9:16" in video_format:
                    format_suffix = "_9x16"
                elif "4:5" in video_format:
                    format_suffix = "_4x5"
                else:
                    format_suffix = ""
                output_path_with_format = base + format_suffix + ext
                # Only the first format generates subtitles
                if first:
                    self.audio_script_pairs.append({
                        "audio": audio_path,
                        "script": script_path,
                        "scene": scene_path,  # Store scene path
                        "output": output_path_with_format,
                        "language": self.LANGUAGES.get(language, "en"),
                        "video_format": video_format,
                        "generate_subtitles": True,
                        "model_size": self.model_size.get(),
                        "use_scene": use_scene_var.get(),  # Store the toggle
                        "animation_style": self.animation_style.get(),  # Store animation style
                        "word_lead_in": self.word_lead_in.get(),  # Store lead-in timing
                        "max_words_per_subtitle": self.max_words_per_subtitle.get()  # Store max words setting
                    })
                    display_str = f"Audio: {os.path.basename(audio_path)} | Script: {os.path.basename(script_path)} | Scene: {os.path.basename(scene_path) if scene_path else 'None'} | Output: {os.path.basename(output_path_with_format)} | Lang: {language} | Model: {self.model_size.get()} | Format: {video_format} (with subtitles) | Scene: {'Yes' if use_scene_var.get() else 'No'} | Animation: {self.animation_style.get()} | Lead-in: {self.word_lead_in.get()}ms | Max Words: {self.max_words_per_subtitle.get()}"
                    first = False
                else:
                    self.audio_script_pairs.append({
                        "audio": audio_path,
                        "script": script_path,
                        "scene": scene_path,  # Store scene path
                        "output": output_path_with_format,
                        "language": self.LANGUAGES.get(language, "en"),
                        "video_format": video_format,
                        "generate_subtitles": False,
                        "model_size": self.model_size.get(),
                        "use_scene": use_scene_var.get(),  # Store the toggle
                        "animation_style": self.animation_style.get(),  # Store animation style
                        "word_lead_in": self.word_lead_in.get(),  # Store lead-in timing
                        "max_words_per_subtitle": self.max_words_per_subtitle.get()  # Store max words setting
                    })
                    display_str = f"Audio: {os.path.basename(audio_path)} | Script: {os.path.basename(script_path)} | Scene: {os.path.basename(scene_path) if scene_path else 'None'} | Output: {os.path.basename(output_path_with_format)} | Lang: {language} | Model: {self.model_size.get()} | Format: {video_format} (video only) | Scene: {'Yes' if use_scene_var.get() else 'No'} | Animation: {self.animation_style.get()} | Lead-in: {self.word_lead_in.get()}ms | Max Words: {self.max_words_per_subtitle.get()}"
                self.pair_listbox.insert(tk.END, display_str)
            lang_window.destroy()
        tk.Button(lang_window, text="OK", command=set_lang).pack(pady=10)

    def batch_process_videos(self):
        if not self.audio_script_pairs:
            messagebox.showerror("Error", "Please add at least one audio/script pair.")
            return
        thread = threading.Thread(target=self._batch_process_videos_thread)
        thread.daemon = True
        thread.start()

    def _batch_process_videos_thread(self):
        total = len(self.audio_script_pairs)
        srt_path_for_batch = None
        for idx, pair in enumerate(self.audio_script_pairs, 1):
            audio_path = pair["audio"]
            script_path = pair["script"]
            output_path = pair["output"]
            language_code = pair["language"]
            video_format = pair["video_format"]
            generate_subtitles = pair.get("generate_subtitles", True)
            model_size = pair.get("model_size", "medium")  # Get model size from pair
            animation_style = pair.get("animation_style", "typewriter")  # Get animation style from pair
            word_lead_in = pair.get("word_lead_in", 100)  # Get lead-in timing from pair
            max_words_per_subtitle = pair.get("max_words_per_subtitle", 8)  # Get max words setting from pair
            self.status_var.set(f"Processing {idx}/{total}: {os.path.basename(audio_path)}")
            self.audio_file.set(audio_path)
            self.script_file.set(script_path)
            self.output_file.set(output_path)
            self.language.set(language_code)
            # Pass srt_path_for_batch to process_single_video
            srt_path_for_batch = self.process_single_video(
                language_code, video_format, generate_subtitles=generate_subtitles, srt_path_for_batch=srt_path_for_batch, model_size=model_size, animation_style=animation_style, word_lead_in=word_lead_in, max_words_per_subtitle=max_words_per_subtitle
            )
        self.status_var.set("Batch processing complete!")
        messagebox.showinfo("Batch Complete", f"Processed {total} videos.")

    # --- In process_single_video, use srt_path_for_batch for all subsequent videos ---
    def process_single_video(self, language_code=None, video_format="16:9 (1920x1080)", generate_subtitles=True, srt_path_for_batch=None, model_size="medium", animation_style="typewriter", word_lead_in=100, max_words_per_subtitle=8):
        try:
            if not generate_subtitles:
                # Use the stored SRT path for all subsequent videos
                if not srt_path_for_batch or not os.path.exists(srt_path_for_batch):
                    raise FileNotFoundError("No subtitle file found for video generation.")
                subtitles = pysrt.open(srt_path_for_batch, encoding='utf-8')
                self.status_var.set("Creating video with subtitles...")
                self.progress_var.set(95)
                self.create_video_with_subtitles(subtitles, show_success_message=False, video_format=video_format, animation_style=animation_style)
                self.progress_var.set(100)
                return srt_path_for_batch
            self.status_var.set("Loading script...")
            self.progress_var.set(10)
            script_lines = self.load_script()
            self.status_var.set("Loading Whisper model...")
            self.progress_var.set(20)
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model(model_size)
            if language_code is None:
                language_code = self.LANGUAGES.get(self.language.get(), "en")
            # Use WhisperX if available
            if WHISPERX_AVAILABLE:
                self.status_var.set("Transcribing audio with WhisperX (better timestamping)...")
                self.progress_var.set(40)
                segments = self._transcribe_with_whisperx(language_code, model_size)
            else:
                self.status_var.set("Transcribing audio with Whisper...")
                self.progress_var.set(40)
                result = self.whisper_model.transcribe(
                    self.audio_file.get(),
                    language=language_code,
                    word_timestamps=True
                )
                segments = result["segments"] if "segments" in result else []
            self.status_var.set("Aligning transcription and grouping by punctuation...")
            self.progress_var.set(70)
            if WHISPERX_AVAILABLE:
                audio_file_path = self.audio_file.get()
                try:
                    lang = language_code
                    if hasattr(segments, 'language'):
                        lang = segments.language
                    import whisperx
                    model_a, metadata = whisperx.load_align_model(language_code=lang, device="cpu")
                    aligned_result = whisperx.align(segments, model_a, metadata, audio_file_path, "cpu", return_char_alignments=False)
                    word_segments = aligned_result.get('word_segments', [])
                    # Update stored word segments with aligned data
                    self.current_word_segments = word_segments.copy()
                except Exception as e:
                    print(f"WhisperX alignment failed: {e}")
                    word_segments = []
            else:
                word_segments = []
                for segment in segments:
                    if 'words' in segment:
                        for w in segment['words']:
                            if w['word'].strip():
                                word_segments.append(w)
                    elif hasattr(segment, 'words') and segment.words:
                        for w in segment.words:
                            if w.word.strip():
                                word_segments.append({'word': w.word, 'start': w.start, 'end': w.end})
                    elif 'text' in segment:
                        text = segment['text']
                        start_time = segment['start']
                        end_time = segment['end']
                        word_list = text.split()
                        if word_list:
                            duration_per_word = (end_time - start_time) / len(word_list)
                            for i, word in enumerate(word_list):
                                word_start = start_time + (i * duration_per_word)
                                word_end = word_start + duration_per_word
                                word_segments.append({'word': word, 'start': word_start, 'end': word_end})

            # --- Apply phrase-level corrections here ---
            word_segments = force_correct_phrases(word_segments, PHRASE_CORRECTIONS)

            # 2. Group words by punctuation
            logical_subtitles = self.group_words_by_punctuation(word_segments, max_words=max_words_per_subtitle)

            # 2.5. ✍️ NEW: Correct known names and errors
            logical_subtitles = correct_known_errors(logical_subtitles, CORRECTIONS)

            # 3. Adjust overlaps
            logical_subtitles = adjust_subtitle_overlaps(logical_subtitles, min_gap_seconds=0.1)

            # 4. ⏱️ NEW: Add padding for better perceived sync
            logical_subtitles = add_subtitle_padding(logical_subtitles)

            # 4.5. Apply lead-in and lead-out for better readability
            audio_duration = self.get_audio_duration()  # Clamp subtitles to audio length
            logical_subtitles = apply_subtitle_lead_in_out(logical_subtitles, lead_in=0.4, lead_out=0.4, audio_duration=audio_duration)

            # 4.6. Filter out subtitles that are empty or only punctuation
            import string
            def is_meaningful_sub(sub):
                text = sub['text'].strip()
                return text and any(c.isalnum() for c in text)
            logical_subtitles = [sub for sub in logical_subtitles if is_meaningful_sub(sub)]

            # 4.7. Merge one-word subtitles with the previous subtitle
            def merge_one_word_subs(subs):
                if not subs:
                    return subs
                merged = [subs[0]]
                for sub in subs[1:]:
                    if len(sub['text'].strip().split()) == 1 and len(merged[-1]['text'].strip().split()) > 0:
                        # Merge with previous
                        merged[-1]['text'] = merged[-1]['text'].rstrip() + ' ' + sub['text'].lstrip()
                        merged[-1]['end'] = sub['end']  # Extend end time
                    else:
                        merged.append(sub)
                return merged
            logical_subtitles = merge_one_word_subs(logical_subtitles)

            # 5. Convert to pysrt.SubRipItem objects
            subtitles = []
            n = len(logical_subtitles)
            for idx, sub in enumerate(logical_subtitles, 1):
                offset = self.timing_offset.get()
                adjusted_start = max(0, sub['start'] + offset)
                # Calculate the default end time
                adjusted_end = max(adjusted_start + 0.5, sub['end'] + offset)
                # Clamp end time to the start of the next subtitle, if any
                if idx < n:
                    next_start = max(0, logical_subtitles[idx]['start'] + offset)
                    if adjusted_end > next_start:
                        adjusted_end = next_start - 0.01  # Small gap
                # Clamp end time to audio duration
                if adjusted_end > audio_duration:
                    adjusted_end = audio_duration
                if adjusted_start > audio_duration:
                    adjusted_start = audio_duration - 0.1  # Prevent start after audio ends
                if adjusted_end <= adjusted_start:
                    adjusted_end = min(audio_duration, adjusted_start + 1.5)
                subtitles.append(
                    pysrt.SubRipItem(
                        index=idx,
                        text=sub['text'],
                        start=self.seconds_to_time(adjusted_start),
                        end=self.seconds_to_time(adjusted_end)
                    )
                )

            self.status_var.set("Fixing subtitle overlaps...")
            self.progress_var.set(85)
            self.fix_subtitle_overlaps(subtitles)

            self.status_var.set("Saving subtitles...")
            self.progress_var.set(90)
            self.save_subtitles(subtitles)
            srt_path = self.output_file.get()
            # Only create video if not subtitles only
            if not self.subtitles_only.get():
                self.status_var.set("Creating video with subtitles...")
                self.progress_var.set(95)
                # In batch mode, set show_success_message to False
                self.create_video_with_subtitles(subtitles, show_success_message=False, video_format=video_format, animation_style=animation_style, word_lead_in=word_lead_in)
            self.progress_var.set(100)
            self.status_var.set("Subtitles generated successfully!" if self.subtitles_only.get() else "Subtitles and video generated successfully!")
            self.update_preview(subtitles)
            return srt_path
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Error in batch video creation: {e}")
            return srt_path_for_batch

def adjust_subtitle_overlaps(subtitle_lines, min_gap_seconds=0.1):
    """
    Adjusts subtitle start times to prevent them from overlapping.

    Args:
        subtitle_lines: A list of subtitle dictionaries.
        min_gap_seconds: The minimum silent gap to enforce between clips.

    Returns:
        The adjusted list of subtitle dictionaries.
    """
    # We start checking from the second clip
    for i in range(1, len(subtitle_lines)):
        previous_line = subtitle_lines[i-1]
        current_line = subtitle_lines[i]

        # Check for overlap
        if current_line['start'] < previous_line['end']:
            # Overlap detected!
            # Move the start time of the current clip to be after the previous one ends
            new_start_time = previous_line['end'] + min_gap_seconds
            
            # Make sure we don't push the start time past the end time
            if new_start_time < current_line['end']:
                print(f"Adjusting overlap: Clip {i+1} start time from {current_line['start']:.2f}s to {new_start_time:.2f}s")
                current_line['start'] = new_start_time
            else:
                # This would make the clip have zero or negative duration, so we just log it.
                # This can happen with very short, rapid-fire words.
                print(f"Warning: Could not fix overlap for clip {i+1} as it would result in a negative duration.")
                
    return subtitle_lines

def correct_known_errors(subtitle_lines, corrections):
    """
    Corrects known transcription errors in the subtitle text.

    Args:
        subtitle_lines: The list of subtitle dictionaries.
        corrections: A dictionary where key is the wrong text
                     and value is the correct text.
    """
    for line in subtitle_lines:
        for wrong, correct in corrections.items():
            # Use .replace() to find and fix errors
            # .casefold() makes the search case-insensitive
            if wrong.casefold() in line['text'].casefold():
                line['text'] = line['text'].replace(wrong, correct)
    return subtitle_lines

# --- NEW: Force phrase-level correction for word segments ---
def force_correct_phrases(word_segments, corrections):
    """
    Finds and replaces entire sequences of incorrect words with a correct phrase,
    preserving the start and end times of the original sequence.

    Args:
        word_segments: The list of word dictionaries from whisperx.align().
        corrections: A dictionary where the key is the incorrect phrase (lowercase)
                     and the value is the correct phrase.
    """
    # Create a new list to store the corrected segments
    corrected_segments = []
    i = 0
    while i < len(word_segments):
        found_match = False
        # Check each correction against the current position in the segments
        for wrong_phrase, correct_phrase in corrections.items():
            wrong_words = wrong_phrase.lower().split()
            num_words = len(wrong_words)

            # Ensure we don't look past the end of the segments list
            if i + num_words > len(word_segments):
                continue

            # Get the sequence of words from the transcription to check for a match
            segment_phrase_words = [word_segments[j]['word'].lower().strip(".,?!") for j in range(i, i + num_words)]
            
            # Check if the sequence of words matches the incorrect phrase
            if segment_phrase_words == wrong_words:
                # Match found!
                print(f"Found and corrected: '{' '.join(segment_phrase_words)}' -> '{correct_phrase}'")
                
                # Create a new, single word segment for the corrected phrase
                new_segment = {
                    'word': correct_phrase,
                    'start': word_segments[i]['start'],   # Start time of the first word
                    'end': word_segments[i + num_words - 1]['end'], # End time of the last word
                    # 'score' is optional, you can leave it out or average it
                }
                corrected_segments.append(new_segment)
                
                # Skip the index ahead by the number of words we just replaced
                i += num_words
                found_match = True
                break # Move to the next position in the segments
        
        # If no correction was applied at this position, just add the original word
        if not found_match:
            corrected_segments.append(word_segments[i])
            i += 1
            
    return corrected_segments

def add_subtitle_padding(subtitle_lines, lead_in_seconds=0.25):
    """
    Adds a small lead-in time to make subtitles feel more responsive.
    """
    for line in subtitle_lines:
        # Subtract from the start time
        new_start = line['start'] - lead_in_seconds
        # Ensure start time doesn't go below zero
        line['start'] = max(0, new_start)
    return subtitle_lines

def apply_subtitle_lead_in_out(subtitle_lines, lead_in=0.25, lead_out=0.25, audio_duration=None):
    """
    Adjusts subtitle timings to appear earlier (lead-in) and stay longer (lead-out),
    without causing overlaps or exceeding audio duration.
    """
    n = len(subtitle_lines)
    for i, sub in enumerate(subtitle_lines):
        # Lead-in: start earlier, but not before 0 or previous end
        new_start = sub['start'] - lead_in
        if i > 0:
            prev_end = subtitle_lines[i-1]['end']
            new_start = max(new_start, prev_end + 0.01)  # Small gap
        sub['start'] = max(0, new_start)
        # Lead-out: end later, but not after next start or audio duration
        new_end = sub['end'] + lead_out
        if i < n - 1:
            next_start = subtitle_lines[i+1]['start']
            new_end = min(new_end, next_start - 0.01)  # Small gap
        if audio_duration is not None:
            new_end = min(new_end, audio_duration)
        sub['end'] = max(sub['start'] + 0.1, new_end)  # Ensure at least 0.1s duration
    return subtitle_lines

def wrap_text_for_format(text, video_format, max_chars_per_line=None):
    """
    Wraps text based on video format to prevent overflow in vertical formats.
    
    Args:
        text: The text to wrap
        video_format: The video format string
        max_chars_per_line: Optional override for max characters per line
    
    Returns:
        The wrapped text with line breaks
    """
    if max_chars_per_line is None:
        # Set character limits based on format
        if "9:16" in video_format or "1080x1920" in video_format:
            max_chars_per_line = 25  # Vertical format - shorter lines
        elif "4:5" in video_format or "1080x1350" in video_format:
            max_chars_per_line = 30  # Square-ish format - medium lines
        else:
            max_chars_per_line = 50  # Horizontal format - longer lines
    
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        # Check if adding this word would exceed the limit
        if len(current_line + " " + word) <= max_chars_per_line:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            # Start a new line
            if current_line:
                lines.append(current_line)
            current_line = word
    
    # Add the last line
    if current_line:
        lines.append(current_line)
    
    return "\n".join(lines)

# --- Typewriter animation helper with multiple styles and improved timing ---
def create_word_typewriter_clip(subtitle_data, video_size, font="Arial-Bold", fontsize=70, color='black', bg_color='transparent', video_format="16:9 (1920x1080)", animation_style="typewriter", word_timings=None, app_instance=None):
    """
    Create subtitle clips with different animation styles and improved word-level timing.
    
    Animation styles:
    - "typewriter": Word-by-word appearance with actual timing
    - "fade_in": Words fade in smoothly at their actual speech time
    - "slide_up": Words slide up from bottom at their actual speech time
    - "bounce": Words bounce in with elastic effect at their actual speech time
    - "glitch": Words appear with glitch effect at their actual speech time
    - "wave": Words appear in wave pattern at their actual speech time
    - "zoom_in": Zoom in effect at actual speech time
    - "typewriter_enhanced": Typewriter with blinking cursor and actual timing
    
    word_timings: List of dicts with 'word', 'start', 'end' for precise timing
    """
    full_text = subtitle_data['text']
    line_duration = subtitle_data['end'] - subtitle_data['start']
    
    # Defensive checks
    if not full_text or not isinstance(full_text, str) or line_duration <= 0:
        return None
    
    # Ensure bg_color is a valid string or None
    if bg_color not in ['white', None, 'transparent']:
        bg_color = None
    
    # Wrap text for vertical formats
    wrapped_text = wrap_text_for_format(full_text, video_format)
    
    # Adjust font size for different formats
    if "9:16" in video_format or "1080x1920" in video_format:
        fontsize = min(fontsize, 50)  # Smaller font for vertical
    elif "4:5" in video_format or "1080x1350" in video_format:
        fontsize = min(fontsize, 60)  # Medium font for square-ish
    elif "16:9" in video_format or "1920x1080" in video_format:
        fontsize = min(fontsize, 70)  # Standard font for horizontal
    
    words = full_text.split()
    if not words:
        return None
    
    # Use actual word timings if available, otherwise fall back to equal distribution
    if word_timings and len(word_timings) == len(words):
        # Create a mapping from word text to timing
        word_timing_map = {}
        for wt in word_timings:
            if wt['word'].strip() in words:
                word_timing_map[wt['word'].strip()] = wt
        
        # Create clips with actual timing
        word_clips = []
        for i, word in enumerate(words):
            displayed_words = words[:i+1]
            displayed_text = wrap_text_for_format(" ".join(displayed_words), video_format)
            
            if not displayed_text or not isinstance(displayed_text, str):
                continue
            
            try:
                # Get actual timing for this word
                if word in word_timing_map:
                    word_timing = word_timing_map[word]
                    word_start = word_timing['start']
                    word_end = word_timing['end']
                    word_duration = word_end - word_start
                    
                    # Add user-configurable lead-in for better perceived sync
                    # Get lead-in from the app instance if available
                    lead_in = 0.1  # Default 100ms
                    try:
                        if hasattr(app_instance, 'word_lead_in'):
                            lead_in = app_instance.word_lead_in.get() / 1000.0  # Convert ms to seconds
                    except:
                        pass
                    
                    adjusted_start = max(0, word_start - lead_in)
                    adjusted_duration = word_duration + lead_in
                else:
                    # Fallback: equal distribution
                    duration_per_word = line_duration / len(words)
                    adjusted_start = subtitle_data['start'] + (i * duration_per_word)
                    adjusted_duration = duration_per_word
                
                # Create base text clip
                word_clip = TextClip(
                    displayed_text,
                    fontsize=fontsize, font=font, color=color, bg_color=bg_color,
                    method='label'  # Use 'label' for tight-fitting text clips
                ).set_duration(adjusted_duration)
                
                # Apply animation style
                if animation_style == "typewriter":
                    # Current style: simple word-by-word with actual timing
                    pass
                    
                elif animation_style == "fade_in":
                    # Smooth fade in effect
                    fade_duration = min(0.3, adjusted_duration * 0.5)
                    word_clip = word_clip.fadein(fade_duration)
                    
                elif animation_style == "slide_up":
                    # Slide up from bottom
                    slide_distance = 50
                    word_clip = word_clip.set_position(('center', 'center'))
                    word_clip = word_clip.set_position(lambda t: ('center', 'center' + slide_distance * (1 - min(1, t/0.3))))
                    
                elif animation_style == "bounce":
                    # Bounce effect with elastic
                    bounce_height = 30
                    word_clip = word_clip.set_position(('center', 'center'))
                    word_clip = word_clip.set_position(lambda t: ('center', 'center' + bounce_height * (1 - min(1, t/0.4)) * (1 - min(1, t/0.4))))
                    
                elif animation_style == "glitch":
                    # Glitch effect with random position shifts
                    import random
                    def glitch_position(t):
                        if t < 0.1:  # First 0.1 seconds
                            x_offset = random.randint(-5, 5)
                            y_offset = random.randint(-3, 3)
                            return ('center' + x_offset, 'center' + y_offset)
                        else:
                            return ('center', 'center')
                    word_clip = word_clip.set_position(glitch_position)
                    
                elif animation_style == "wave":
                    # Wave pattern appearance
                    wave_amplitude = 20
                    wave_frequency = 2
                    word_clip = word_clip.set_position(('center', 'center'))
                    word_clip = word_clip.set_position(lambda t: ('center', 'center' + wave_amplitude * (1 - min(1, t/0.3)) * (1 - min(1, t/0.3)) * (1 - min(1, t/0.3))))
                    
                elif animation_style == "zoom_in":
                    # Zoom in effect
                    scale_factor = 1.5
                    word_clip = word_clip.set_position(('center', 'center'))
                    word_clip = word_clip.resize(lambda t: scale_factor - (scale_factor - 1) * min(1, t/0.3))
                    
                elif animation_style == "typewriter_enhanced":
                    # Enhanced typewriter with cursor effect
                    cursor_char = "|"
                    if i == len(words) - 1:  # Last word
                        displayed_text += cursor_char
                    word_clip = TextClip(
                        displayed_text,
                        fontsize=fontsize, font=font, color=color, bg_color=bg_color,
                        method='label'
                    ).set_duration(adjusted_duration)
                
                # Set the start time for this word clip
                word_clip = word_clip.set_start(adjusted_start)
                word_clips.append(word_clip)
                
            except Exception as e:
                print(f"Warning: Could not create TextClip for text '{displayed_text}': {e}")
                continue
        
        if not word_clips:
            return None
        
        # For actual timing, we want to show the final text for the full duration
        # but with individual word animations at their correct times
        final_clip = word_clips[-1].set_duration(line_duration)
        
    else:
        # Fallback to original equal distribution method
        word_clips = []
        duration_per_word = line_duration / len(words)
        
        for i in range(len(words)):
            displayed_words = words[:i+1]
            displayed_text = wrap_text_for_format(" ".join(displayed_words), video_format)
            
            if not displayed_text or not isinstance(displayed_text, str):
                continue
            
            try:
                # Create base text clip
                word_clip = TextClip(
                    displayed_text,
                    fontsize=fontsize, font=font, color=color, bg_color=bg_color,
                    method='label'  # Use 'label' for tight-fitting text clips
                ).set_duration(duration_per_word)
                
                # Apply animation style (same as above)
                if animation_style == "typewriter":
                    pass
                elif animation_style == "fade_in":
                    fade_duration = min(0.3, duration_per_word * 0.5)
                    word_clip = word_clip.fadein(fade_duration)
                elif animation_style == "slide_up":
                    slide_distance = 50
                    word_clip = word_clip.set_position(('center', 'center'))
                    word_clip = word_clip.set_position(lambda t: ('center', 'center' + slide_distance * (1 - min(1, t/0.3))))
                elif animation_style == "bounce":
                    bounce_height = 30
                    word_clip = word_clip.set_position(('center', 'center'))
                    word_clip = word_clip.set_position(lambda t: ('center', 'center' + bounce_height * (1 - min(1, t/0.4)) * (1 - min(1, t/0.4))))
                elif animation_style == "glitch":
                    import random
                    def glitch_position(t):
                        if t < 0.1:
                            x_offset = random.randint(-5, 5)
                            y_offset = random.randint(-3, 3)
                            return ('center' + x_offset, 'center' + y_offset)
                        else:
                            return ('center', 'center')
                    word_clip = word_clip.set_position(glitch_position)
                elif animation_style == "wave":
                    wave_amplitude = 20
                    word_clip = word_clip.set_position(('center', 'center'))
                    word_clip = word_clip.set_position(lambda t: ('center', 'center' + wave_amplitude * (1 - min(1, t/0.3)) * (1 - min(1, t/0.3)) * (1 - min(1, t/0.3))))
                elif animation_style == "zoom_in":
                    scale_factor = 1.5
                    word_clip = word_clip.set_position(('center', 'center'))
                    word_clip = word_clip.resize(lambda t: scale_factor - (scale_factor - 1) * min(1, t/0.3))
                elif animation_style == "typewriter_enhanced":
                    cursor_char = "|"
                    if i == len(words) - 1:
                        displayed_text += cursor_char
                    word_clip = TextClip(
                        displayed_text,
                        fontsize=fontsize, font=font, color=color, bg_color=bg_color,
                        method='label'
                    ).set_duration(duration_per_word)
                
                word_clips.append(word_clip)
                
            except Exception as e:
                print(f"Warning: Could not create TextClip for text '{displayed_text}': {e}")
                continue
        
        if not word_clips:
            return None
        
        # Combine all word clips for equal distribution
        if animation_style in ["slide_up", "bounce", "glitch", "wave", "zoom_in"]:
            final_clip = word_clips[-1].set_duration(line_duration)
        else:
            final_clip = concatenate_videoclips(word_clips)
    
    return final_clip.set_position('center').set_start(subtitle_data['start'])

# --- Animation style selector function ---
def get_animation_style_options():
    """Get list of available animation styles with descriptions"""
    return {
        "typewriter": "Classic word-by-word appearance",
        "fade_in": "Smooth fade in effect",
        "slide_up": "Words slide up from bottom",
        "bounce": "Bouncy elastic entrance",
        "glitch": "Digital glitch effect",
        "wave": "Wave pattern appearance",
        "zoom_in": "Zoom in effect",
        "typewriter_enhanced": "Typewriter with blinking cursor"
    }

# NOTE: Please copy magick.exe from your ImageMagick install directory to C:\ImageMagick if not already done.

def main():
    root = tk.Tk()
    app = AdvancedSubtitleGeneratorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 