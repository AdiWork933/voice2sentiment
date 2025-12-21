import librosa
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("E://Audio_D//emotion_model.keras") # Model path

EMOTIONS = {
    0: 'angry', 
    1: 'disgust', 
    2: 'fear', 
    3: 'happy', 
    4: 'neutral', 
    5: 'sad'
}

def predict_audio(file_path):
    # Same parameters as training
    SAMPLE_RATE = 22050
    DURATION = 3 
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

    try:
        # Load and Preprocess
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        else:
            padding = SAMPLES_PER_TRACK - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, SAMPLES_PER_TRACK - len(y) - offset), 'constant')

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = mfcc.T
        
        # Reshape for the model (1, Time_Steps, Features)
        mfcc = mfcc[np.newaxis, ...] 

        # Predict
        prediction = model.predict(mfcc)
        predicted_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        
        return EMOTIONS[predicted_index], confidence

    except Exception as e:
        return f"Error: {e}", 0.0

if __name__ == "__main__":
    # Replace with path to a test file
    test_file = "E://Audio_D//hindi//sad//H_A_S_0003.wav" # audio path for testing 
    
    emotion, conf = predict_audio(test_file)
    print(f"Predicted Emotion: {emotion} (Confidence: {conf*100:.2f}%)")
