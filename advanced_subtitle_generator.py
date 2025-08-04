import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pydub import AudioSegment
# Explicitly set the path to FFmpeg if it's not in the system's PATH
# Assumes FFmpeg was installed to C:\\ffmpeg as instructed.
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"
import pysrt
import os
import threading
import json
from datetime import datetime, timedelta
import re
import difflib
import whisper
from moviepy.editor import TextClip, CompositeVideoClip, concatenate_videoclips, ColorClip, AudioFileClip
import moviepy.config as mpy_config
mpy_config.change_settings({"IMAGEMAGICK_BINARY": r"C:\\ImageMagick\\magick.exe"})

# Set ImageMagick binary to a path without spaces for MoviePy compatibility
os.environ["IMAGEMAGICK_BINARY"] = r"C:\\ImageMagick\\magick.exe"  # <-- Make sure magick.exe is copied here

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
try:
    import subprocess
    
    # Try to find ImageMagick in PATH
    result = subprocess.run(['magick', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        # Get the full path
        result = subprocess.run(['where', 'magick'], capture_output=True, text=True)
        if result.returncode == 0:
            magick_path = result.stdout.strip().split('\n')[0]
            print(f"MoviePy configured to use ImageMagick at: {magick_path}")
except Exception as e:
    print(f"Could not configure MoviePy ImageMagick path: {e}")

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
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
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
        preview_frame = ttk.LabelFrame(main_frame, text="Generated Subtitles Preview", padding="10")
        preview_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(20, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=15, width=80)
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        main_frame.rowconfigure(6, weight=1)
        
        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="Model Settings", padding="10")
        model_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        ttk.Label(model_frame, text="Whisper Model Size:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_size, 
                                  values=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], 
                                  state="readonly", width=15)
        model_combo.grid(row=0, column=1, sticky=tk.W)
        model_combo.set("medium")
        
        # Model info
        model_info = "tiny (39MB) < base (74MB) < small (244MB) < medium (769MB) < large (1550MB) < large-v2 (1550MB) < large-v3 (1550MB)"
        ttk.Label(model_frame, text=model_info, foreground='gray', wraplength=600).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Subtitles only checkbox
        subtitles_only_cb = ttk.Checkbutton(main_frame, text="Subtitles only (do not create video)", variable=self.subtitles_only)
        subtitles_only_cb.grid(row=8, column=0, columnspan=3, pady=(5, 0), sticky=tk.W)
        
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
            logical_subtitles = self.group_words_by_punctuation(word_segments)

            # 2.5. ✍️ NEW: Correct known names and errors
            logical_subtitles = correct_known_errors(logical_subtitles, CORRECTIONS)

            # 3. Adjust overlaps
            logical_subtitles = adjust_subtitle_overlaps(logical_subtitles, min_gap_seconds=0.1)

            # 4. ⏱️ NEW: Add padding for better perceived sync
            logical_subtitles = add_subtitle_padding(logical_subtitles)

            # 4.5. Apply lead-in and lead-out for better readability
            audio_duration = self.get_audio_duration()  # Clamp subtitles to audio length
            logical_subtitles = apply_subtitle_lead_in_out(logical_subtitles, lead_in=0.25, lead_out=0.25, audio_duration=audio_duration)

            # 4.6. Filter out subtitles that are empty or only punctuation
            import string
            def is_meaningful_sub(sub):
                text = sub['text'].strip()
                return text and any(c.isalnum() for c in text)
            logical_subtitles = [sub for sub in logical_subtitles if is_meaningful_sub(sub)]

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
                # Simple and robust chunking for long sentences
                chunks = [" ".join(words[j:j+max_words]) for j in range(0, len(words), max_words)]
                # If the last chunk is very short, merge it with the previous
                if len(chunks) > 1 and len(chunks[-1].split()) <= 2:
                    chunks[-2] = chunks[-2] + " " + chunks[-1]
                    chunks = chunks[:-1]
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
                initial_prompt=prompt_text,
                vad_options={"vad_onset": 0.3, "vad_offset": 0.2},
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
        
    def group_words_by_punctuation(self, word_segments, max_words=10, merge_short_lines=True):
        """
        Groups word segments into subtitle lines, now with a max word count.
        """
        subtitle_lines = []
        current_line_words = []

        for word_data in word_segments:
            word = word_data['word'].strip()
            current_line_words.append(word_data)

            # Check for sentence-ending punctuation
            if word.endswith(('.', '?', '!')):
                # Logic to create a line (same as before)
                start_time = current_line_words[0]['start']
                end_time = current_line_words[-1]['end']
                text = " ".join([w['word'] for w in current_line_words])
                subtitle_lines.append({"text": text, "start": start_time, "end": end_time})
                current_line_words = []
            # Check for commas on lines that aren't too short
            elif word.endswith(','):
                if merge_short_lines and len(current_line_words) <= 2:
                    pass # Don't break on short comma lines
                else:
                    # Logic to create a line (same as before)
                    start_time = current_line_words[0]['start']
                    end_time = current_line_words[-1]['end']
                    text = " ".join([w['word'] for w in current_line_words])
                    subtitle_lines.append({"text": text, "start": start_time, "end": end_time})
                    current_line_words = []
            # 💡 NEW: Check if the line exceeds the max word count
            elif len(current_line_words) >= max_words:
                # Logic to create a line
                start_time = current_line_words[0]['start']
                end_time = current_line_words[-1]['end']
                text = " ".join([w['word'] for w in current_line_words])
                subtitle_lines.append({"text": text, "start": start_time, "end": end_time})
                current_line_words = []


        # Handle any remaining words
        if current_line_words:
            start_time = current_line_words[0]['start']
            end_time = current_line_words[-1]['end']
            text = " ".join([w['word'] for w in current_line_words])
            subtitle_lines.append({"text": text, "start": start_time, "end": end_time})

        return subtitle_lines
    
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

    def create_video_with_subtitles(self, subtitles, show_success_message=True, video_format="16:9 (1920x1080)"):
        """Create a simple video with a white background and black subtitle text, synchronized to the audio."""
        print("=== Starting video creation process ===")
        try:
            # Check if ImageMagick is available for text rendering
            print("Checking ImageMagick availability...")
            try:
                # Try to create a simple text clip to test ImageMagick
                test_text = TextClip("TEST", fontsize=20, color='black')
                test_text.close()
                imagemagick_available = True
                print("✓ ImageMagick is working correctly")
            except Exception as e:
                if "ImageMagick" in str(e):
                    imagemagick_available = False
                    print("✗ ImageMagick not available")
                    messagebox.showwarning("ImageMagick Required", 
                        "ImageMagick is not installed, which is required for text rendering.\n\n"
                        "To install ImageMagick:\n"
                        "1. Download from: https://imagemagick.org/script/download.php#windows\n"
                        "2. Install with default settings\n"
                        "3. Restart this application\n\n"
                        "For now, only the SRT subtitle file will be created.")
                    return
                else:
                    imagemagick_available = True
                    print(f"⚠ ImageMagick test failed with: {e}")
            
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
                    typewriter_clip = create_word_typewriter_clip(sub_data, size, video_format=video_format)
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
                            typewriter_clip_scene = create_word_typewriter_clip(sub_data_scene, size, font="Arial-Bold", fontsize=70 if size[1] >= 1080 else 50, color='black', bg_color='white', video_format=video_format)
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
                            typewriter_clip_white = create_word_typewriter_clip(sub_data_white, size, font="Arial-Bold", fontsize=70 if size[1] >= 1080 else 50, color='black', bg_color=None, video_format=video_format)
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
                        typewriter_clip_white = create_word_typewriter_clip(sub_data_white, size, font="Arial-Bold", fontsize=70 if size[1] >= 1080 else 50, color='black', bg_color=None, video_format=video_format)
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
                        "use_scene": use_scene_var.get()  # Store the toggle
                    })
                    display_str = f"Audio: {os.path.basename(audio_path)} | Script: {os.path.basename(script_path)} | Scene: {os.path.basename(scene_path) if scene_path else 'None'} | Output: {os.path.basename(output_path_with_format)} | Lang: {language} | Model: {self.model_size.get()} | Format: {video_format} (with subtitles) | Scene: {'Yes' if use_scene_var.get() else 'No'}"
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
                        "use_scene": use_scene_var.get()  # Store the toggle
                    })
                    display_str = f"Audio: {os.path.basename(audio_path)} | Script: {os.path.basename(script_path)} | Scene: {os.path.basename(scene_path) if scene_path else 'None'} | Output: {os.path.basename(output_path_with_format)} | Lang: {language} | Model: {self.model_size.get()} | Format: {video_format} (video only) | Scene: {'Yes' if use_scene_var.get() else 'No'}"
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
            self.status_var.set(f"Processing {idx}/{total}: {os.path.basename(audio_path)}")
            self.audio_file.set(audio_path)
            self.script_file.set(script_path)
            self.output_file.set(output_path)
            self.language.set(language_code)
            # Pass srt_path_for_batch to process_single_video
            srt_path_for_batch = self.process_single_video(
                language_code, video_format, generate_subtitles=generate_subtitles, srt_path_for_batch=srt_path_for_batch, model_size=model_size
            )
        self.status_var.set("Batch processing complete!")
        messagebox.showinfo("Batch Complete", f"Processed {total} videos.")

    # --- In process_single_video, use srt_path_for_batch for all subsequent videos ---
    def process_single_video(self, language_code=None, video_format="16:9 (1920x1080)", generate_subtitles=True, srt_path_for_batch=None, model_size="medium"):
        try:
            if not generate_subtitles:
                # Use the stored SRT path for all subsequent videos
                if not srt_path_for_batch or not os.path.exists(srt_path_for_batch):
                    raise FileNotFoundError("No subtitle file found for video generation.")
                subtitles = pysrt.open(srt_path_for_batch, encoding='utf-8')
                self.status_var.set("Creating video with subtitles...")
                self.progress_var.set(95)
                self.create_video_with_subtitles(subtitles, show_success_message=False, video_format=video_format)
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
            logical_subtitles = self.group_words_by_punctuation(word_segments)

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
                self.create_video_with_subtitles(subtitles, show_success_message=False, video_format=video_format)
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

# --- Typewriter animation helper ---
# --- Always center subtitles in create_word_typewriter_clip ---
def create_word_typewriter_clip(subtitle_data, video_size, font="Arial-Bold", fontsize=70, color='black', bg_color='transparent', video_format="16:9 (1920x1080)"):
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
        fontsize = min(fontsize, 70)  # Standard font for horizontal, but still apply wrapping
    words = full_text.split()
    if not words:
        return None
    word_clips = []
    duration_per_word = line_duration / len(words)
    for i in range(len(words)):
        displayed_words = words[:i+1]
        displayed_text = wrap_text_for_format(" ".join(displayed_words), video_format)
        if not displayed_text or not isinstance(displayed_text, str):
            continue
        try:
            word_clip = TextClip(
                displayed_text,
                fontsize=fontsize, font=font, color=color, bg_color=bg_color,
                method='label' # Use 'label' for tight-fitting text clips
            ).set_duration(duration_per_word)
            word_clips.append(word_clip)
        except Exception as e:
            print(f"Warning: Could not create TextClip for text '{displayed_text}': {e}")
            continue
    if not word_clips:
        return None
    typewriter_sequence = concatenate_videoclips(word_clips)
    return typewriter_sequence.set_position('center').set_start(subtitle_data['start'])

# NOTE: Please copy magick.exe from your ImageMagick install directory to C:\ImageMagick if not already done.

def main():
    root = tk.Tk()
    app = AdvancedSubtitleGeneratorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 