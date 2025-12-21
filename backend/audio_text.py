import whisper

# Load the model
model = whisper.load_model("small") 

# Your corrected path
audio_file = r"D:\Audio_D\bengali\happy\B_A_H_0200.wav"

# --- THE FIX IS HERE ---
# Add 'fp16=False' to fix the "nan" error on GTX 1650
result = model.transcribe(audio_file, fp16=False)

# Print the transcription
print(result["text"])