import os
import time
import numpy as np
import tensorflow as tf
from api_config import preprocess_audio

# --- Emotion Mapping ---
EMOTIONS = {
    0: 'angry', 1: 'disgust', 2: 'fear', 
    3: 'happy', 4: 'neutral', 5: 'sad'
}

class EmotionPredictor:
    """Handles loading and inference for the Emotion Detection model."""
    
    def __init__(self, model_path):
        self.model = None
        self._load_model(model_path)

    def _load_model(self, path):
        print("--- 📂 Loading Emotion Model (CPU Optimized) ---")
        if path and os.path.exists(path):
            try:
                self.model = tf.keras.models.load_model(path, compile=False)
                print("✅ Emotion model loaded")
            except Exception as e:
                print(f"❌ Error loading emotion model: {e}")
        else:
            print(f"⚠️ Emotion model path invalid or missing")
        print("-" * 35)

    def predict(self, audio_input):
        """Runs audio through the emotion model."""
        start_time = time.time() # Start timer
        
        if self.model is None:
            return None, 0.0, 0.0
            
        features = preprocess_audio(audio_input)
        if features is None:
            return None, 0.0, 0.0

        predictions = self.model.predict(features, verbose=0)[0]
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions)
        
        predicted_emotion = EMOTIONS.get(predicted_index, "Unknown")
        
        prediction_time = round(time.time() - start_time, 4) # Stop timer
        
        return predicted_emotion, confidence, prediction_time
