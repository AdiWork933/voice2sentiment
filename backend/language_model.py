import librosa
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import load_model

# --- Global Configuration ---
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
MFCC_COUNT = 40
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
                # compile=False makes loading much faster
                model = load_model(path, compile=False)
                return lang, model
            except Exception as e:
                print(f"❌ Error loading {lang}: {e}")
                return lang, None
        else:
            print(f"⚠️ File missing for {lang}: {path}")
            return lang, None

    def _load_models_parallel(self):
        print("--- 🚀 Loading Multi-Language Models (Parallel) ---")
        
        # Use 3 workers since we have 3 models
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

    def _extract_features(self, audio_input):
        try:
            y, sr = librosa.load(audio_input, sr=SAMPLE_RATE)

            if len(y) > SAMPLES_PER_TRACK:
                y = y[:SAMPLES_PER_TRACK]
            else:
                padding = int(SAMPLES_PER_TRACK - len(y))
                offset = padding // 2
                y = np.pad(y, (offset, padding - offset), 'constant')

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_COUNT)
            features = mfcc.T
            return features[np.newaxis, ...]

        except Exception as e:
            print(f"Feature Extraction Error: {e}")
            return None

    def predict(self, audio_input):
        """Runs the audio through all loaded binary models."""
        X_predict = self._extract_features(audio_input)

        if X_predict is None:
            return "Error during feature extraction."

        best_match = {'language': 'Undetermined', 'probability': 0.0}

        # Inference is fast enough to keep sequential
        for lang, model in self.models.items():
            if model is None:
                continue

            predictions = model.predict(X_predict, verbose=0)
            probability = predictions[0][0]
            is_positive = (probability >= THRESHOLD)

            if is_positive and probability > best_match['probability']:
                best_match['language'] = lang.capitalize()
                best_match['probability'] = probability

        return best_match['language']
