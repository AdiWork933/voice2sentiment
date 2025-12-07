import librosa
import numpy as np
import tensorflow as tf
import os

# ==========================================
# 1. Configuration (Must match Training Config)
# ==========================================
MODEL_SAVE_PATH = "language_detection_model.keras"  # Model path

# Defined Languages (Mapping from ID back to name)
ID_TO_LANGUAGE = {
    0: 'Hindi',
    1: 'English',
    2: 'Bengali'
}

# Audio Parameters (Must match Training Config)
SAMPLE_RATE = 22050
DURATION = 3 # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# ==========================================
# 2. Preprocessing Function
# ==========================================
def preprocess_audio(file_path):
    """
    Loads an audio file and converts it into the MFCC feature format 
    expected by the trained 1D CNN model.                                                                                                             
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Padding or Truncating to ensure fixed length
        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        else:
            padding = SAMPLES_PER_TRACK - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, SAMPLES_PER_TRACK - len(y) - offset), 'constant')

        # Extract MFCCs (40 features, same as training)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Transpose to shape (Time, Feats) -> (130, 40)
        feature = mfcc.T
        
        # Add a batch dimension: (Time, Feats) -> (1, Time, Feats)
        # This is required because the Keras model expects a batch input.
        feature = np.expand_dims(feature, axis=0) 
        
        return feature
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ==========================================
# 3. Prediction Function
# ==========================================
def predict_language(model, audio_file_path):
    """
    Makes a prediction on a single audio file using the loaded model.
    """
    # 1. Preprocess the audio
    input_features = preprocess_audio(audio_file_path)
    
    if input_features is None:
        return "Prediction Failed (Feature extraction error)"

    # 2. Make prediction
    # model.predict returns probabilities for each class (e.g., [0.9, 0.05, 0.05])
    probabilities = model.predict(input_features, verbose=0)[0] 
    
    # 3. Get the predicted class ID (the one with the highest probability)
    predicted_id = np.argmax(probabilities)
    
    # 4. Map the ID back to the language name
    predicted_language = ID_TO_LANGUAGE.get(predicted_id, "Unknown Language ID")
    confidence = probabilities[predicted_id] * 100
    
    # Return both the result and the confidence score
    return predicted_language, confidence, probabilities

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    
    # A. Check if model exists
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model file '{MODEL_SAVE_PATH}' not found.")
        print("Please run the training script first to create the saved model.")
        exit()

    # B. Load the trained model
    print(f"Loading model from {MODEL_SAVE_PATH}...")
    try:
        loaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        print("Model loaded successfully. Ready for prediction.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit()

    print("---")
    
    # Testing audio path
    TEST_FILE_PATH = "bengali/angry/B_A_A_0001.wav" 
    
    
    if os.path.exists(TEST_FILE_PATH):
        language, confidence, probs = predict_language(loaded_model, TEST_FILE_PATH)

        print(f"File: {os.path.basename(TEST_FILE_PATH)}")
        print(f"✅ Predicted Language: **{language}**")
        print(f"Confidence: {confidence:.2f}%")
        print("\nFull Probabilities:")
        # Display all probabilities for verification
        for id_val, name in ID_TO_LANGUAGE.items():
            print(f"- {name}: {probs[id_val]*100:.2f}%")
    else:
        print(f"Error: Test file not found at '{TEST_FILE_PATH}'. Please update the path and try again.")
