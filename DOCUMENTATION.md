================================================================================
            Voice2Sentimental - Complete Technical Documentation
================================================================================

Project: Voice2Sentimental - Unified Audio & Text Analysis Platform
Author: Aditya Choudhary
Description: Full-stack AI application for audio emotion detection, 
             language identification, and speech transcription.

================================================================================
                              TABLE OF CONTENTS
================================================================================

1.  PROJECT OVERVIEW
2.  SYSTEM ARCHITECTURE
3.  TECHNOLOGY STACK
4.  FILE STRUCTURE & DESCRIPTIONS
5.  BACKEND DOCUMENTATION
    5.1 Configuration (api_config.py)
    5.2 Emotion Detection (emotion_model.py)
    5.3 Language Detection (language_model.py)
    5.4 Audio Transcription (audio_text_model.py)
    5.5 API Routes (main_api.py)
6.  FRONTEND DOCUMENTATION
    6.1 Flask Application (app.py)
    6.2 Templates
7.  MODEL ARCHITECTURE DETAILS
8.  API ENDPOINT SPECIFICATIONS
9.  ENVIRONMENT SETUP
10. DEPLOYMENT GUIDE
11. TROUBLESHOOTING

================================================================================
                          1. PROJECT OVERVIEW
================================================================================

Voice2Sentimental is a dual-service AI platform that processes audio files
for three primary purposes:

1. EMOTION DETECTION: Classifies speech into 6 emotional states
   - Classes: angry, disgust, fear, happy, neutral, sad
   - Model: TensorFlow CNN trained on audio MFCC features

2. LANGUAGE IDENTIFICATION: Detects spoken language
   - Supported: Hindi, English, Bengali
   - Approach: Ensemble of 3 binary classifiers

3. SPEECH TRANSCRIPTION: Converts audio to text
   - Model: Faster-Whisper (OpenAI Whisper implementation)
   - GPU-accelerated with automatic language detection

4. TEXT REFINEMENT: Post-processing with Google Gemini
   - Grammar correction
   - Educational formatting (study guide style)

================================================================================
                          2. SYSTEM ARCHITECTURE
================================================================================

                         +---------------------+
                         |   Client Browser    |
                         +----------+----------+
                                    |
                                    v
                         +---------------------+
                         |   Flask Frontend    |
                         |   (Port 5000)       |
                         |   - User Auth       |
                         |   - File Upload     |
                         |   - Results Display |
                         +----------+----------+
                                    |
                                    | HTTP Requests
                                    v
                         +---------------------+
                         |   FastAPI Backend   |
                         |   (Port 8000)       |
                         +----------+----------+
                                    |
                +-------------------+-------------------+
                |                   |                   |
                v                   v                   v
       +----------------+  +----------------+  +----------------+
       | Emotion Model  |  | Language Model |  |   Whisper      |
       | TensorFlow/CPU |  | TensorFlow/CPU |  |  Faster-Whisper|
       | .keras format  |  | 3 binary models|  |  GPU/CUDA      |
       +----------------+  +----------------+  +----------------+
                |                   |                   |
                v                   v                   v
       +----------------+  +----------------+  +----------------+
       | 6-class CNN      |  | One-vs-Rest    |  | large-v3-turbo |
       | MFCC features    |  | classifiers    |  | VAD filter     |
       +----------------+  +----------------+  +----------------+

WORKLOAD DISTRIBUTION:
- CPU: TensorFlow models (emotion + language) - avoids GPU memory competition
- GPU: Faster-Whisper transcription - benefits from CUDA acceleration

================================================================================
                          3. TECHNOLOGY STACK
================================================================================

BACKEND:
- FastAPI 0.104.1         - High-performance async web framework
- Uvicorn 0.24.0          - ASGI server
- TensorFlow              - Deep learning framework
- PyTorch 2.3.1 (CUDA)    - For Whisper backend
- Faster-Whisper          - Optimized Whisper implementation
- Librosa 0.10.1          - Audio processing & MFCC extraction
- Pydub                   - Audio format conversion
- Google GenAI            - Gemini API client
- Python-dotenv           - Environment variable management

FRONTEND:
- Flask                   - Web framework
- Jinja2                  - Template engine
- Werkzeug                - Security & WSGI utilities
- Requests                - HTTP client for API calls

TRAINING/DEV:
- NumPy 1.26.2            - Numerical computing
- SoundFile               - Audio I/O
- Audioread               - Backend audio decoder

INFRASTRUCTURE:
- FFmpeg                  - Audio codec/conversion
- CUDA 12.1               - GPU acceleration
- Cloudflared             - Tunnel for external access (optional)

================================================================================
                       4. FILE STRUCTURE & DESCRIPTIONS
================================================================================

ROOT DIRECTORY:
├── README.md                     - Project overview and quick start
├── DOCUMENTATION.txt             - This comprehensive documentation file
├── requirements.txt              - Python dependencies
├── .env                          - Environment variables (gitignored)
├── .gitignore                    - Git exclusion rules
├── run.ps1                       - PowerShell launcher script
├── run.txt                       - Quick command reference
└── cloudflared-windows-amd64.exe - Tunnel utility (optional)

BACKEND DIRECTORY (backend/):
├── main_api.py                   - FastAPI application entry point
│                                 - API route definitions
│                                 - Key rotation logic
│                                 - HTTP Basic Auth for docs
│
├── api_config.py                 - Shared configuration
│                                 - TensorFlow CPU restriction
│                                 - MFCC preprocessing pipeline
│                                 - Audio constants (SR=22050, duration=3s)
│
├── emotion_model.py              - EmotionPredictor class
│                                 - Model loading from .keras file
│                                 - 6-class prediction logic
│                                 - Confidence scoring
│
├── language_model.py             - MultiLanguagePredictor class
│                                 - Parallel model loading (ThreadPool)
│                                 - 3 binary classifier ensemble
│                                 - Threshold-based voting (0.5)
│
├── audio_text_model.py           - AudioTextPredictor class
│                                 - Faster-Whisper initialization
│                                 - GPU configuration (int8_float16)
│                                 - VAD-filtered transcription
│
├── evaluate_dataset.py           - Model evaluation utilities
│                                 - Batch processing
│                                 - Performance metrics
│                                 - Results CSV export
│
├── evaluation_results.csv        - Stored evaluation metrics
├── time_analysis_dashboard.png   - Performance visualization
│
└── models/                       - Trained ML model files
    ├── emotion_model.keras                   (7.4 MB)
    ├── hindi_vs_nonhindi_detection_model.keras       (7.4 MB)
    ├── english_vs_nonenglish_detection_model.keras   (7.4 MB)
    ├── bengali_vs_nonbengali_detection_model.keras     (7.4 MB)
    └── language_detection_model.keras                (7.4 MB)

FRONTEND DIRECTORY (frontend/):
├── app.py                        - Flask application
│                                 - User authentication system
│                                 - Session management
│                                 - API client (concurrent calls)
│                                 - Route handlers
│
├── users.json                    - User credentials storage
│                                 - JSON format with hashed passwords
│
├── favicon/                      - Browser icon assets
│
└── templates/                    - Jinja2 HTML templates
    ├── login.html                - User login page
    ├── register.html             - User registration page
    └── index.html                - Main dashboard (analysis interface)

ROW CODES DIRECTORY (row codes/):
├── model.py                      - Qwen LLM integration reference
│
├── audio_text.py                 - Legacy audio text utilities
│
├── backend/                      - Alternative API implementations
│   ├── main_api.py              - Extended API with additional endpoints
│   ├── api_config.py            - Extended configuration
│   └── language_model.py        - Alternative language detection
│
├── frontend/                     - Frontend reference code
│
├── UI/                           - UI component references
│
├── hindi_train.py                - Hindi model training script
├── english_train.py              - English model training script
└── bengali_train.py              - Bengali model training script

================================================================================
                         5. BACKEND DOCUMENTATION
================================================================================

5.1 CONFIGURATION (api_config.py)
---------------------------------
Purpose: Central configuration and audio preprocessing

Key Constants:
- SAMPLE_RATE = 22050          - Audio resampling rate
- DURATION = 3                  - Fixed analysis window (seconds)
- SAMPLES_PER_TRACK = 66150     - 22050 * 3
- N_MFCC = 40                   - Number of Mel-frequency cepstral coefficients

TensorFlow GPU Fix:
```python
tf.config.set_visible_devices([], 'GPU')  # Force CPU for TF
```
This prevents TensorFlow from consuming GPU memory, leaving it available
for Faster-Whisper.

Preprocessing Pipeline (preprocess_audio function):
1. Load audio: librosa.load(audio_source, sr=SAMPLE_RATE)
2. Truncate: Clip to SAMPLES_PER_TRACK if longer
3. Pad: Center-pad with zeros if shorter
4. Extract MFCC: librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
5. Transpose: mfcc.T for time-series format
6. Reshape: Add batch dimension [1, time, features]

Output shape: (1, 259, 40) where 259 = 22050*3/256 (hop length)

5.2 EMOTION DETECTION (emotion_model.py)
------------------------------------------
Class: EmotionPredictor

Emotion Mapping:
{
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad'
}

Methods:
- __init__(model_path): Loads Keras model, compiles=False for inference only
- predict(audio_input): Returns (emotion, confidence, prediction_time)

Prediction Logic:
1. Extract MFCC features via preprocess_audio
2. model.predict(features, verbose=0) - suppress output
3. np.argmax for class selection
4. np.max for confidence extraction
5. Round time.time() difference to 4 decimal places

5.3 LANGUAGE DETECTION (language_model.py)
--------------------------------------------
Class: MultiLanguagePredictor

Architecture: Ensemble of 3 binary classifiers
- hindi_vs_nonhindi_detection_model.keras
- english_vs_nonenglish_detection_model.keras
- bengali_vs_nonbengali_detection_model.keras

Threshold: 0.5 (positive if probability >= 0.5)

Loading Strategy:
```python
with ThreadPoolExecutor(max_workers=3) as executor:
    # Load all 3 models concurrently
```
This reduces startup time from ~3s sequential to ~1.5s parallel.

Prediction Logic:
1. Extract MFCC features
2. For each loaded model:
   - Run inference: model.predict(features, verbose=0)
   - Get probability: predictions[0][0]
   - Check threshold: is_positive = probability >= 0.5
3. Select winner: Highest probability among positive predictions
4. Return capitalized language name or "Undetermined"

5.4 AUDIO TRANSCRIPTION (audio_text_model.py)
---------------------------------------------
Class: AudioTextPredictor

Model: Faster-Whisper large-v3-turbo

GPU Configuration:
- device="cuda"                 - Use NVIDIA GPU
- compute_type="int8_float16"   - Mixed precision for 4GB VRAM
- Alternative: "int8" for safer memory usage

Transcription Features:
- beam_size=2                   - Balance speed vs accuracy
- vad_filter=True               - Skip silent audio segments

Output Tuple:
(
    full_text.strip(),          - Cleaned transcribed text
    prediction_time,            - Processing duration (seconds)
    info.language,              - Detected language code (e.g., 'en', 'hi')
    info.language_probability   - Confidence score (0.0 - 1.0)
)

5.5 API ROUTES (main_api.py)
----------------------------
Class: KeyManager
- Manages rotation of multiple Gemini API keys
- get_current_key(): Returns active key
- rotate_key(): Cycles to next key on failure

Security:
- HTTPBasic for /docs and /openapi.json endpoints
- Credentials from DOCS_USERNAME and DOCS_PASSWORD env vars
- secrets.compare_digest() for timing-attack-safe comparison

Lifespan Manager:
- @asynccontextmanager async def lifespan(app)
- Loads all models on startup
- Yields control during app runtime
- Cleanup on shutdown

Routes:

1. GET /
   Returns status info including GPU configuration
   
2. POST /predict_emotion
   Request: multipart/form-data with audio_file
   Response: {emotion, confidence, prediction_time}
   
3. POST /M_predict_language
   Request: multipart/form-data with audio_file
   Response: {predicted_language, prediction_time}
   
4. POST /predict_text
   Request: multipart/form-data with audio_file
   Response: {transcribed_text, detected_language, language_probability, prediction_time}
   
5. POST /refine-text
   Request: JSON {"text": "input text here"}
   Response: {"processed_text": "refined output"}
   
   - Uses Gemini 1.5 Flash
   - Key rotation on failures
   - System prompt for educational vs casual formatting

6. GET /docs (protected)
   Swagger UI documentation
   
7. GET /openapi.json (protected)
   OpenAPI schema

================================================================================
                        6. FRONTEND DOCUMENTATION
================================================================================

6.1 FLASK APPLICATION (app.py)
------------------------------
Configuration:
- SECRET_KEY: From env for session encryption
- BASE_URL: Backend API endpoint (default: Cloudflare tunnel)
- USERS_FILE: 'users.json' for credential storage

Authentication System:
- Password hashing: werkzeug.security.generate_password_hash
- Verification: check_password_hash
- Session-based: flask.session

Routes:

1. GET/POST /login
   - Template: login.html
   - Validates credentials against users.json
   - Sets session['logged_in'] = True on success
   
2. GET/POST /register
   - Template: register.html
   - Password confirmation validation
   - Username uniqueness check
   - Hashed password storage
   
3. GET /logout
   - Clears session
   - Redirects to login
   
4. GET / (index)
   - Protected by @login_required
   - Template: index.html
   - Displays main analysis dashboard

API Client Functions:

1. call_api(url, audio_bytes, filename)
   - Makes POST request to backend
   - 120 second timeout for large files
   - Returns JSON or error dict

2. audio_only() - Route: POST /audio_only
   - Parallel API calls using ThreadPoolExecutor
   - Calls both /M_predict_language and /predict_emotion simultaneously
   - Returns combined results
   
3. transcribe_only() - Route: POST /transcribe_only
   - Single API call to /predict_text
   - Returns transcription data

6.2 TEMPLATES
-------------

login.html:
- Simple login form
- Username and password fields
- Error message display

register.html:
- Registration form
- Username, password, confirm_password fields
- Password match validation
- Success/error message display

index.html:
- Main dashboard interface
- Audio file upload interface
- Analysis type selection
- Results display area
- User logout button

================================================================================
                        7. MODEL ARCHITECTURE DETAILS
================================================================================

7.1 EMOTION MODEL ARCHITECTURE
------------------------------
Input: (batch_size, 259, 40, 1) - MFCC spectrogram

Typical CNN Structure:
- Conv2D(32, kernel_size=(3,3), activation='relu')
- MaxPooling2D(pool_size=(2,2))
- Conv2D(64, kernel_size=(3,3), activation='relu')
- MaxPooling2D(pool_size=(2,2))
- Conv2D(128, kernel_size=(3,3), activation='relu')
- MaxPooling2D(pool_size=(2,2))
- Flatten()
- Dense(128, activation='relu')
- Dropout(0.5)
- Dense(6, activation='softmax')  # 6 emotions

Output: Probability distribution over 6 emotion classes

Training Data: Likely RAVDESS or similar emotional speech dataset

7.2 LANGUAGE MODEL ARCHITECTURE
-------------------------------
3 Separate Binary Classifiers:

Each follows same architecture:
Input: (batch_size, 259, 40, 1)

Typical Structure:
- Conv2D(32, (3,3), activation='relu')
- MaxPooling2D((2,2))
- Conv2D(64, (3,3), activation='relu')
- MaxPooling2D((2,2))
- Flatten()
- Dense(64, activation='relu')
- Dropout(0.3)
- Dense(1, activation='sigmoid')  # Binary output

Training Approach:
- Hindi model: Hindi (1) vs All others (0)
- English model: English (1) vs All others (0)
- Bengali model: Bengali (1) vs All others (0)

Inference:
- Run all 3 models
- Collect probabilities
- Apply threshold (0.5)
- Select highest confident positive

7.3 WHISPER ARCHITECTURE
------------------------
Model: large-v3-turbo
- Encoder-Decoder Transformer
- 1.5B parameters (turbo variant)
- Multilingual (99 languages)
- Optimized for speed with minimal accuracy loss

Quantization:
- int8_float16 = 8-bit weights, 16-bit activations
- Reduces VRAM from ~10GB to ~4GB
- Suitable for GTX 1650 (4GB)

VAD (Voice Activity Detection):
- Silero VAD integrated
- Automatically skips silent segments
- 30% faster processing on average

================================================================================
                        8. API ENDPOINT SPECIFICATIONS
================================================================================

8.1 DETAILED ENDPOINT DOCS
--------------------------

Health Check
------------
GET /

Response:
{
    "status": "active",
    "docs_url": "/docs",
    "gpu_status": "GPU running Whisper, CPU running TensorFlow",
    "Build by": "Aditya Choudhary"
}

Emotion Prediction
------------------
POST /predict_emotion
Content-Type: multipart/form-data

Parameters:
- audio_file: File (required) - Audio file (wav, mp3, etc.)

Success Response (200):
{
    "emotion": "happy",
    "confidence": "92.45%",
    "prediction_time": "0.3421 seconds"
}

Error Responses:
- 400: Invalid audio format or FFmpeg missing
- 503: Emotion model not loaded

Language Prediction
-------------------
POST /M_predict_language
Content-Type: multipart/form-data

Parameters:
- audio_file: File (required)

Success Response (200):
{
    "predicted_language": "English",
    "prediction_time": "0.2847 seconds"
}

Error Responses:
- 400: Invalid audio format
- 503: Predictor not ready

Speech Transcription
--------------------
POST /predict_text
Content-Type: multipart/form-data

Parameters:
- audio_file: File (required)

Success Response (200):
{
    "transcribed_text": "Hello, this is a test transcription",
    "detected_language": "en",
    "language_probability": "98.32%",
    "prediction_time": "1.2456 seconds"
}

Error Responses:
- 400: Audio processing failed
- 503: Text model not loaded

Text Refinement
---------------
POST /refine-text
Content-Type: application/json

Request Body:
{
    "text": "your text to refine here"
}

Success Response (200):
{
    "processed_text": "Refined and formatted text output"
}

Error Responses:
- 400: Invalid JSON or missing text field
- 500: All API keys failed

================================================================================
                          9. ENVIRONMENT SETUP
================================================================================

9.1 COMPLETE .env EXAMPLE
-------------------------

# ==========================================
# BACKEND CONFIGURATION
# ==========================================

# Model Paths (adjust based on your structure)
EMOTION_MODEL_PATH="backend/models/emotion_model.keras"
HINDI_MODEL_PATH="backend/models/hindi_vs_nonhindi_detection_model.keras"
ENGLISH_MODEL_PATH="backend/models/english_vs_nonenglish_detection_model.keras"
BENGALI_MODEL_PATH="backend/models/bengali_vs_nonbengali_detection_model.keras"

# Google Gemini API Keys (min 1, max 6)
# Get keys from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY_1=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
GEMINI_API_KEY_2=AIzaSyYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
GEMINI_API_KEY_3=
GEMINI_API_KEY_4=
GEMINI_API_KEY_5=
GEMINI_API_KEY_6=

# API Documentation Credentials
DOCS_USERNAME=admin
DOCS_PASSWORD=your_secure_password_here

# ==========================================
# FRONTEND CONFIGURATION
# ==========================================

# Backend API URL
API=http://127.0.0.1:8000
# Or for cloud deployment:
# API=https://your-app.trycloudflare.com

# Flask Secret Key (generate random string)
SECRET_KEY=your_random_secret_key_minimum_16_chars

# Optional: Default admin credentials for initial setup
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123

9.2 WINDOWS SETUP STEPS
-----------------------

1. Install Python 3.10+ from python.org

2. Install FFmpeg:
   - Download from https://ffmpeg.org/download.html
   - Extract to C:\ffmpeg
   - Add C:\ffmpeg\bin to System PATH
   - Verify: ffmpeg -version

3. Install CUDA 12.1 (for GPU):
   - Download from NVIDIA developer site
   - Required for Whisper GPU acceleration

4. Open PowerShell as Administrator:
   cd d:\audio_api_backend
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

5. Create .env file with your configuration

6. Run the application:
   .\run.ps1

9.3 LINUX/MAC SETUP STEPS
--------------------------

1. Install dependencies:
   sudo apt update
   sudo apt install python3.10 python3-pip ffmpeg

2. For Mac with Homebrew:
   brew install python ffmpeg

3. Setup virtual environment:
   cd audio_api_backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

4. Create .env file

5. Run manually:
   # Terminal 1
   cd backend
   uvicorn main_api:app --reload --port 8000

   # Terminal 2
   cd frontend
   python app.py

================================================================================
                         10. DEPLOYMENT GUIDE
================================================================================

10.1 LOCAL DEVELOPMENT
----------------------
Use run.ps1 for Windows:
- Starts backend on port 8000
- Opens new terminal for frontend on port 5000
- Both share the same virtual environment

10.2 CLOUDFLARE TUNNEL (Quick Public URL)
------------------------------------------
For temporary public access:

1. Download cloudflared (included: cloudflared-windows-amd64.exe)

2. Start your backend locally on port 8000

3. Create tunnel for frontend (port 5000):
   .\cloudflared-windows-amd64.exe tunnel --url http://127.0.0.1:5000

4. Share the generated URL (e.g., https://folks-arg-cached-feedback.trycloudflare.com)

Note: Update frontend's BASE_URL in .env to the tunnel URL

10.3 PRODUCTION DEPLOYMENT CHECKLIST
--------------------------------------

Security:
- [ ] Change default DOCS_USERNAME and DOCS_PASSWORD
- [ ] Generate strong SECRET_KEY
- [ ] Use HTTPS in production
- [ ] Add rate limiting (use slowapi or similar)
- [ ] Validate all file uploads (size, type)

Performance:
- [ ] Use gunicorn instead of Flask dev server
- [ ] Add Redis for session storage
- [ ] Implement request caching
- [ ] Use CDN for static files

Monitoring:
- [ ] Add logging to file
- [ ] Set up health check endpoint
- [ ] Monitor GPU/CPU usage
- [ ] Track API key rotation events

================================================================================
                          11. TROUBLESHOOTING
================================================================================

ISSUE: "FFmpeg not found" error
--------------------------------
Solution:
1. Download FFmpeg from https://ffmpeg.org
2. Extract to a permanent location (e.g., C:\ffmpeg)
3. Add bin folder to System PATH
4. Restart terminal and verify: ffmpeg -version

ISSUE: "CUDA out of memory" error
----------------------------------
Solution:
1. In audio_text_model.py, change compute_type:
   From: compute_type="int8_float16"
   To:   compute_type="int8"

2. Or reduce Whisper model size:
   From: model_size="large-v3-turbo"
   To:   model_size="medium"

ISSUE: TensorFlow using GPU instead of CPU
-------------------------------------------
Solution:
Check api_config.py has this code at the top:
```python
try:
    tf.config.set_visible_devices([], 'GPU')
    print("TensorFlow restricted to CPU.")
except RuntimeError as e:
    print(e)
```

ISSUE: Import errors for torch/torchaudio
------------------------------------------
Solution:
Reinstall with CUDA support:
pip uninstall torch torchvision torchaudio
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

ISSUE: Gemini API key failures
-------------------------------
Solution:
1. Verify keys are valid at https://makersuite.google.com
2. Check .env file has correct key format
3. Ensure at least one GEMINI_API_KEY_1 is set
4. Check internet connectivity
5. Review logs for specific error messages

ISSUE: Frontend cannot connect to backend
-----------------------------------------
Solution:
1. Verify backend is running on port 8000
2. Check .env API variable points to correct URL
3. For local: API=http://127.0.0.1:8000
4. Check firewall/antivirus is not blocking
5. Try accessing http://127.0.0.1:8000 in browser

ISSUE: Slow audio processing
----------------------------
Solutions:
1. Enable GPU for Whisper (check nvidia-smi)
2. Enable VAD filter (already on by default)
3. Reduce audio file size before upload
4. Check CPU/GPU temperatures (throttling?)

ISSUE: Permission denied on PowerShell
---------------------------------------
Solution:
Run this command before executing scripts:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

================================================================================
                            END OF DOCUMENTATION
================================================================================

For questions or issues, refer to:
- README.md for quick reference
- FastAPI docs: http://localhost:8000/docs (when running)
- Flask app: http://localhost:5000 (when running)

Built by Aditya Choudhary
