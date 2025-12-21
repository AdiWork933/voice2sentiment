import librosa
import numpy as np
import os
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
    """

    def __init__(self, model_paths):
        """
        Initializes the predictor by loading the three language models.
        :param model_paths: Dictionary {'hindi': path, 'english': path, 'bengali': path}
        """
        self.model_paths = model_paths
        self.models = {}
        self._load_models()

    def _load_models(self):
        print("--- Loading Multi-Language Binary Models ---")
        for lang, path in self.model_paths.items():
            if path and os.path.exists(path):
                try:
                    # Compile=False is faster for inference
                    self.models[lang] = load_model(path, compile=False)
                    print(f"✅ Loaded {lang.capitalize()} model")
                except Exception as e:
                    print(f"❌ Error loading {lang}: {e}")
                    self.models[lang] = None
            else:
                print(f"⚠️ Model file missing for {lang}: {path}")
                self.models[lang] = None
        print("-" * 35)

    def _extract_features(self, audio_input):
        """
        Extracts MFCC features.
        :param audio_input: Can be a file path (str) or a file-like object (BytesIO).
        """
        try:
            y, sr = librosa.load(audio_input, sr=SAMPLE_RATE)

            if len(y) > SAMPLES_PER_TRACK:
                y = y[:SAMPLES_PER_TRACK]
            else:
                padding = SAMPLES_PER_TRACK - len(y)
                offset = padding // 2
                y = np.pad(y, (offset, SAMPLES_PER_TRACK - len(y) - offset), 'constant')

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_COUNT)
            features = mfcc.T
            return features[np.newaxis, ...]

        except Exception as e:
            print(f"Feature Extraction Error: {e}")
            return None

    def predict(self, audio_input):
        """
        Runs the audio through all three binary models.
        :param audio_input: Path to .wav or BytesIO object.
        """
        # 1. Extract features once
        X_predict = self._extract_features(audio_input)

        if X_predict is None:
            return "Error during feature extraction."

        best_match = {'language': 'Undetermined', 'probability': 0.0}

        # 2. Run sequential prediction
        for lang, model in self.models.items():
            if model is None:
                continue

            predictions = model.predict(X_predict, verbose=0)
            probability = predictions[0][0]
            is_positive = (probability >= THRESHOLD)

            # Update best match if this language is a match AND has higher probability
            if is_positive and probability > best_match['probability']:
                best_match['language'] = lang.capitalize()
                best_match['probability'] = probability

        return best_match['language']

# --- Main Execution for Testing ---
if __name__ == "__main__":
    MODEL_PATHS = {
        'hindi': "models/hindi_vs_nonhindi_detection_model.keras",
        'english': "models/english_vs_nonenglish_detection_model.keras",
        'bengali': "models/bengali_vs_nonbengali_detection_model.keras"
    }
    # Test with a dummy file path if needed
    # predictor = MultiLanguagePredictor(MODEL_PATHS)
    pass