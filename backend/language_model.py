import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from api_config import preprocess_audio

# --- Global Configuration ---
THRESHOLD = 0.5

class MultiLanguagePredictor:
    """
    A class to load and sequentially run three binary language classification models.
    Optimized for Parallel Loading.
    """

    def __init__(self, model_paths):
        self.model_paths = model_paths
        self.models = {}
        self._load_models_parallel()

    def _load_single_model(self, lang, path):
        """Helper function to load a single model (worker task)."""
        if path and os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path, compile=False)
                return lang, model
            except Exception as e:
                print(f"❌ Error loading {lang}: {e}")
                return lang, None
        else:
            print(f"⚠️ File missing for {lang}: {path}")
            return lang, None

    def _load_models_parallel(self):
        print("--- 🚀 Loading Multi-Language Models (Parallel) ---")
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_lang = {
                executor.submit(self._load_single_model, lang, path): lang
                for lang, path in self.model_paths.items()
            }
            
            for future in future_to_lang:
                lang, model = future.result()
                if model:
                    self.models[lang] = model
                    print(f"✅ Loaded {lang.capitalize()}")
                else:
                    self.models[lang] = None
        print("-" * 35)

    def predict(self, audio_input):
        """Runs the audio through all loaded binary models."""
        start_time = time.time() # Start timer
        
        features = preprocess_audio(audio_input)

        if features is None:
            return "Error during feature extraction.", 0.0

        best_match = {'language': 'Undetermined', 'probability': 0.0}

        for lang, model in self.models.items():
            if model is None:
                continue

            predictions = model.predict(features, verbose=0)
            probability = predictions[0][0]
            is_positive = (probability >= THRESHOLD)

            if is_positive and probability > best_match['probability']:
                best_match['language'] = lang.capitalize()
                best_match['probability'] = probability

        prediction_time = round(time.time() - start_time, 4) # Stop timer

        return best_match['language'], prediction_time
