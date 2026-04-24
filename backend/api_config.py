import os
import numpy as np
import librosa
from dotenv import load_dotenv
import tensorflow as tf

# --- OPTIMIZATION: SPLIT WORKLOADS (TF on CPU, Whisper on GPU) ---
# Hide the GPU from TensorFlow so it only uses the CPU
try:
    tf.config.set_visible_devices([], 'GPU')
    print("🚦 Traffic Control: TensorFlow restricted to CPU.")
except RuntimeError as e:
    print(e)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration Constants ---
SAMPLE_RATE = 22050
DURATION = 3  # seconds
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
N_MFCC = 40  # Number of MFCC features

def preprocess_audio(audio_source):
    """
    Converts raw audio (bytes or path) into the required MFCC feature array.
    Shared by both Emotion and Language models.
    """
    try:
        y, sr = librosa.load(audio_source, sr=SAMPLE_RATE)

        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        else:
            padding = int(SAMPLES_PER_TRACK - len(y))
            offset = padding // 2
            y = np.pad(y, (offset, padding - offset), 'constant')

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        feature = mfcc.T
        return feature[np.newaxis, ...]

    except Exception as e:
        print(f"Audio Preprocessing Error: {e}")
        return None