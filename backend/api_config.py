import os
import numpy as np
import librosa
from dotenv import load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging clutter

import tensorflow as tf

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration Constants ---
SAMPLE_RATE = 22050
DURATION = 3  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 40  # Number of MFCC features

# Emotion Model Mapping
EMOTIONS = {
    0: 'angry', 1: 'disgust', 2: 'fear', 
    3: 'happy', 4: 'neutral', 5: 'sad'
}

# Language Model Mapping (Single Model)
ID_TO_LANGUAGE = {
    0: 'Hindi', 1: 'English', 2: 'Bengali'
}

# --- Model Paths ---
EMOTION_MODEL_PATH = os.getenv("EMOTION_MODEL_PATH")
LANGUAGE_MODEL_PATH = os.getenv("LANGUAGE_MODEL_PATH")

# --- Global Variables for Models ---
emotion_model = None
language_model = None

def load_models():
    """Loads the single-task Keras models."""
    global emotion_model, language_model

    print("--- 📂 Loading Standard API Models (CPU Optimized) ---")

    # Load Emotion Model
    if EMOTION_MODEL_PATH and os.path.exists(EMOTION_MODEL_PATH):
        try:
            emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH, compile=False)
            print(f"✅ Emotion model loaded")
        except Exception as e:
            print(f"❌ Error loading emotion model: {e}")
    else:
        print(f"⚠️ Emotion model path invalid or missing")

    # Load Language Model
    if LANGUAGE_MODEL_PATH and os.path.exists(LANGUAGE_MODEL_PATH):
        try:
            language_model = tf.keras.models.load_model(LANGUAGE_MODEL_PATH, compile=False)
            print(f"✅ Language model loaded")
        except Exception as e:
            print(f"❌ Error loading language model: {e}")
    else:
        print(f"⚠️ Language model path invalid or missing")

    print("-" * 35)

def preprocess_audio(audio_source):
    """
    Converts raw audio (bytes or path) into the required MFCC feature array.
    """
    try:
        # 1. Decode audio
        y, sr = librosa.load(audio_source, sr=SAMPLE_RATE)

        # 2. Padding/Truncating
        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        else:
            padding = int(SAMPLES_PER_TRACK - len(y))
            offset = padding // 2
            y = np.pad(y, (offset, padding - offset), 'constant')

        # 3. Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

        # 4. Reshape for model (1, Time, Feats)
        feature = mfcc.T
        feature = feature[np.newaxis, ...]

        return feature

    except Exception as e:
        print(f"Audio Preprocessing Error: {e}")
        return None
