import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import speech_recognition as sr
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

class SubtitleGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Subtitle Generator")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.script_file = tk.StringVar()
        self.audio_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        
        # Initialize recognizer
        self.recognizer = sr.Recognizer()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Subtitle Generator", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Script file selection
        ttk.Label(main_frame, text="Script File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        script_entry = ttk.Entry(main_frame, textvariable=self.script_file, width=50)
        script_entry.grid(row=1, column=1, sticky="ew", padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_script).grid(row=1, column=2, pady=5)
        
        # Audio file selection
        ttk.Label(main_frame, text="Audio File:").grid(row=2, column=0, sticky=tk.W, pady=5)
        audio_entry = ttk.Entry(main_frame, textvariable=self.audio_file, width=50)
        audio_entry.grid(row=2, column=1, sticky="ew", padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_audio).grid(row=2, column=2, pady=5)
        
        # Output file selection
        ttk.Label(main_frame, text="Output File:").grid(row=3, column=0, sticky=tk.W, pady=5)
        output_entry = ttk.Entry(main_frame, textvariable=self.output_file, width=50)
        output_entry.grid(row=3, column=1, sticky="ew", padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=3, column=2, pady=5)
        
        # Progress bar
        ttk.Label(main_frame, text="Progress:").grid(row=4, column=0, sticky=tk.W, pady=(20, 5))
        progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        progress_bar.grid(row=4, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=(20, 5))
        
        # Status label
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=5, column=0, columnspan=3, pady=5)
        
        # Generate button
        generate_btn = ttk.Button(main_frame, text="Generate Subtitles", 
                                 command=self.generate_subtitles, style='Accent.TButton')
        generate_btn.grid(row=6, column=0, columnspan=3, pady=20)
        
        # Preview area
        preview_frame = ttk.LabelFrame(main_frame, text="Generated Subtitles Preview", padding="10")
        preview_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(20, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=15, width=80)
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        
        # Configure main frame row weights
        main_frame.rowconfigure(7, weight=1)
        
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
            
            # Load script
            script_lines = self.load_script()
            
            self.status_var.set("Processing audio...")
            self.progress_var.set(30)
            
            # Process audio and generate timestamps
            subtitles = self.process_audio_with_script(script_lines)
            
            self.status_var.set("Saving subtitles...")
            self.progress_var.set(80)
            
            # Save subtitles
            self.save_subtitles(subtitles)
            
            self.progress_var.set(100)
            self.status_var.set("Subtitles generated successfully!")
            
            # Update preview
            self.update_preview(subtitles)
            
            messagebox.showinfo("Success", "Subtitles generated successfully!")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
    def load_script(self):
        """Load and parse the script file"""
        with open(self.script_file.get(), 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into lines and clean up
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        return lines
        
    def process_audio_with_script(self, script_lines):
        """Process audio and align with script lines"""
        # Load audio file
        audio = AudioSegment.from_file(self.audio_file.get())
        
        # Calculate approximate duration per line
        total_duration = len(audio) / 1000  # Convert to seconds
        duration_per_line = total_duration / len(script_lines)
        
        subtitles = []
        current_time = 0
        
        for i, line in enumerate(script_lines):
            # Calculate start and end times
            start_time = current_time
            end_time = start_time + duration_per_line
            
            # Create subtitle entry
            subtitle = pysrt.SubRipItem(
                index=i+1,
                start=self.seconds_to_time(start_time),
                end=self.seconds_to_time(end_time),
                text=line
            )
            subtitles.append(subtitle)
            
            current_time = end_time
            
        return subtitles
        
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

def main():
    root = tk.Tk()
    app = SubtitleGeneratorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 