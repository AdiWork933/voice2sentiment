import os
import numpy as np
import librosa
from dotenv import load_dotenv

# --- OPTIMIZATION: FORCE CPU FOR AUDIO MODELS ---
# Faster inference for small audio models vs moving data to GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
        return feature[np.newaxis, ...]

    except Exception as e:
        print(f"Audio Preprocessing Error: {e}")
        return None
