import time
from faster_whisper import WhisperModel

class AudioTextPredictor:
    """Handles loading and inference for the Voice-to-Text model on the GPU."""
    
    def __init__(self, model_size="large-v3-turbo"):
        print(f"--- 🚀 Loading Faster-Whisper ({model_size}) on GPU ---")
        try:
            # device="cuda" targets your GTX 1650
            # compute_type="int8_float16" compresses the model to fit inside 4GB VRAM
            self.model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
            print("✅ Audio-to-Text model loaded on GPU")
        except Exception as e:
            print(f"❌ Error loading audio-to-text model: {e}")
            self.model = None
        print("-" * 35)

    def predict(self, audio_source):
        """Runs audio through the Whisper model."""
        if self.model is None:
            return "Error: Model not loaded.", 0.0, "unknown", 0.0

        start_time = time.time()
        
        # vad_filter=True instantly skips silent parts of the audio for faster processing
        segments, info = self.model.transcribe(audio_source, beam_size=2, vad_filter=True)
        
        # Generator processes chunks; we loop to extract the full text
        full_text = " ".join([segment.text for segment in segments])
        
        prediction_time = round(time.time() - start_time, 4)
        
        return full_text.strip(), prediction_time, info.language, info.language_probability


# import time
# from faster_whisper import WhisperModel

# class AudioTextPredictor:
#     """Handles loading and inference for the Voice-to-Text model on the GPU."""
    
#     def __init__(self, model_size="large-v3-turbo"):
#         print(f"--- 🚀 Loading Faster-Whisper ({model_size}) on GPU ---")
#         try:
#             # device="cuda" targets your GTX 1650
#             # compute_type="int8" compresses the model to fit safely inside 4GB VRAM
#             self.model = WhisperModel(model_size, device="cuda", compute_type="int8")
#             print("✅ Audio-to-Text model loaded on GPU")
#         except Exception as e:
#             print(f"❌ Error loading audio-to-text model: {e}")
#             self.model = None
#         print("-" * 35)

#     def predict(self, audio_source):
#         """Runs audio through the Whisper model."""
#         if self.model is None:
#             return "Error: Model not loaded.", 0.0, "unknown", 0.0

#         start_time = time.time()
        
#         # vad_filter skips silence, beam_size=1 and condition_on_previous_text=False maximize speed
#         segments, info = self.model.transcribe(
#             audio_source, 
#             beam_size=1, 
#             vad_filter=True,
#             condition_on_previous_text=False
#         )
        
#         # Generator processes chunks; we loop to extract the full text
#         full_text = " ".join([segment.text for segment in segments])
        
#         prediction_time = round(time.time() - start_time, 4)
        
#         return full_text.strip(), prediction_time, info.language, info.language_probability