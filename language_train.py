import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Setup
# ==========================================
DATASET_PATH = "E://Audio_D"  # Your root folder containing 'hindi', 'english', 'bengali' folders
MODEL_SAVE_PATH = "language_detection_model.keras"

# Defined Languages (Target Classes)
LANGUAGES = {
    'hindi': 0,
    'english': 1,
    'bengali': 2
}

# Audio Parameters
SAMPLE_RATE = 22050
DURATION = 3 # Duration in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# ==========================================
# 2. Feature Extraction Function (MFCC)
# ==========================================
def extract_features(file_path):
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

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Transpose to shape (Time, Feats)
        return mfcc.T 
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ==========================================
# 3. Data Loading
# ==========================================
def load_data(dataset_path):
    X = []
    y = []
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Directory {dataset_path} not found!")
    
    print("Starting Feature Extraction for Language Detection...")
    
    # Iterate through the 3 languages
    for lang_name, lang_id in LANGUAGES.items():
        lang_path = os.path.join(dataset_path, lang_name)
        
        if not os.path.isdir(lang_path):
            print(f"Warning: Folder {lang_path} not found, skipping.")
            continue
            
        print(f"Processing Language: {lang_name} (ID: {lang_id})")
        
        # os.walk will find files even if they are inside emotion sub-folders
        for root, dirs, files in os.walk(lang_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    
                    # Extract features
                    feature = extract_features(file_path)
                    
                    if feature is not None:
                        X.append(feature)
                        y.append(lang_id) # Assign the LANGUAGE ID, not emotion

    return np.array(X), np.array(y)

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    # A. Load Dataset
    X, y = load_data(DATASET_PATH)
    
    if len(X) == 0:
        print("No data found. Please check your folder structure. It should be E://Audio_D/hindi, etc.")
        exit()

    print(f"\nData Loaded. Features Shape: {X.shape}") 
    # Shape: (Num_Samples, Time_Steps, MFCC_Features)

    # B. Prepare Data
    # One-hot encode labels for 3 classes
    y_cat = to_categorical(y, num_classes=3)

    # Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    # C. Build Deep Neural Network (1D CNN)
    model = Sequential()

    # Layer 1
    model.add(Conv1D(64, kernel_size=3, padding='same', input_shape=(X.shape[1], X.shape[2])))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Layer 2
    model.add(Conv1D(128, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Layer 3
    model.add(Conv1D(128, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # Flatten and Dense
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    # Output Layer -> CHANGED to 3 for Languages
    model.add(Dense(3)) 
    model.add(Activation('softmax'))

    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # D. Train
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    print("\nStarting Training...")
    history = model.fit(
        X_train, y_train, 
        batch_size=32, 
        epochs=50, 
        validation_data=(X_test, y_test), 
        callbacks=[checkpoint, early_stop]
    )

    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # E. Plot Results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Language Detection Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.legend()
    plt.show()
