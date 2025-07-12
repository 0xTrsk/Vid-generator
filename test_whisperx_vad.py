import whisperx
import torch

# --- SETUP ---
HF_TOKEN = "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"  # Paste your Hugging Face token here
device = "cuda" if torch.cuda.is_available() else "cpu"
audio_file = "path/to/your_audio.wav"  # <-- Replace with your audio file path

try:
    # 1. Load a modern VAD model using your token
    print("Loading VAD model...")
    vad_model = whisperx.load_vad_model(device, use_auth_token=HF_TOKEN)

    # 2. Load the Whisper model as usual
    print("Loading Whisper model...")
    model = whisperx.load_model("medium", device, compute_type="float16")

    # 3. Transcribe, passing the loaded VAD model
    print(f"Transcribing audio file: {audio_file}...")
    result = model.transcribe(audio_file, vad_model=vad_model)
    print("Transcription complete.")

    # 4. Align the transcription (no changes needed here)
    print("Aligning transcription...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio_file, device, return_char_alignments=False)
    print("Alignment complete.")

    # Now 'result["word_segments"]' contains accurate, word-level timestamps
    print("\n--- Timestamps ---")
    for segment in result["word_segments"]:
        print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['word']}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("\nTroubleshooting:")
    print("1. Did you replace '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' with your actual Hugging Face token?")
    print("2. Is your internet connection working to download the models?")
    print("3. Ensure you have the latest versions of whisperx, torch, and pyannote.audio installed in a clean environment.") 