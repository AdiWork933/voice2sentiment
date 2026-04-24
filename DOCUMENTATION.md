# Voice2Sentimental - Complete Technical Documentation

**Project:** Voice2Sentimental - Unified Audio & Text Analysis Platform  
**Author:** Aditya Choudhary  
**Description:** Full-stack AI application for audio emotion detection, language identification, and speech transcription.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [File Structure](#4-file-structure)
5. [Backend Documentation](#5-backend-documentation)
6. [Frontend Documentation](#6-frontend-documentation)
7. [Model Architecture](#7-model-architecture)
8. [API Specifications](#8-api-specifications)
9. [Environment Setup](#9-environment-setup)
10. [Deployment Guide](#10-deployment-guide)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Project Overview

Voice2Sentimental is a dual-service AI platform that processes audio files for four primary purposes:

### 1.1 Emotion Detection
- **Classes:** angry, disgust, fear, happy, neutral, sad
- **Model:** TensorFlow CNN trained on audio MFCC features
- **Runtime:** CPU (TensorFlow)

### 1.2 Language Identification
- **Supported:** Hindi, English, Bengali
- **Approach:** Ensemble of 3 binary classifiers
- **Runtime:** CPU (TensorFlow)

### 1.3 Speech Transcription
- **Model:** Faster-Whisper (OpenAI Whisper implementation)
- **Hardware:** GPU-accelerated with CUDA
- **Features:** Automatic language detection, VAD filtering

### 1.4 Text Refinement
- **Service:** Google Gemini 1.5 Flash
- **Features:** Grammar correction, educational formatting
- **Fallback:** API key rotation on failures

---

## 2. System Architecture

```
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
```

### Workload Distribution
- **CPU:** TensorFlow models (emotion + language) - avoids GPU memory competition
- **GPU:** Faster-Whisper transcription - benefits from CUDA acceleration

---

## 3. Technology Stack

### Backend
| Package | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.104.1 | Async web framework |
| Uvicorn | 0.24.0 | ASGI server |
| TensorFlow | Latest | Deep learning |
| PyTorch | 2.3.1+cu121 | Whisper backend |
| Faster-Whisper | Latest | Optimized speech recognition |
| Librosa | 0.10.1 | Audio processing |
| Pydub | Latest | Audio conversion |
| Google GenAI | Latest | Gemini API client |

### Frontend
| Package | Purpose |
|---------|---------|
| Flask | Web framework |
| Jinja2 | Template engine |
| Werkzeug | Security & WSGI |
| Requests | HTTP client |

### Infrastructure
- **FFmpeg** - Audio codec/conversion
- **CUDA 12.1** - GPU acceleration
- **Cloudflared** - Tunnel for external access

---

## 4. File Structure

```
audio_api_backend/
├── README.md                     # Project overview
├── DOCUMENTATION.md              # This file
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables
├── .gitignore                    # Git exclusions
├── run.ps1                       # PowerShell launcher
├── run.txt                       # Command reference
├── cloudflared-windows-amd64.exe # Tunnel utility
│
├── backend/                      # FastAPI application
│   ├── main_api.py              # API routes & entry point
│   ├── api_config.py            # Configuration & preprocessing
│   ├── emotion_model.py         # Emotion prediction class
│   ├── language_model.py        # Language detection class
│   ├── audio_text_model.py      # Whisper transcription class
│   ├── evaluate_dataset.py      # Evaluation utilities
│   ├── evaluation_results.csv   # Performance metrics
│   ├── time_analysis_dashboard.png
│   └── models/                  # Trained ML models
│       ├── emotion_model.keras
│       ├── hindi_vs_nonhindi_detection_model.keras
│       ├── english_vs_nonenglish_detection_model.keras
│       ├── bengali_vs_nonbengali_detection_model.keras
│       └── language_detection_model.keras
│
├── frontend/                     # Flask application
│   ├── app.py                   # Flask app & auth
│   ├── users.json               # User credentials
│   ├── favicon/
│   └── templates/               # HTML templates
│       ├── login.html
│       ├── register.html
│       └── index.html
```

---

## 5. Backend Documentation

### 5.1 Configuration (`api_config.py`)

**Key Constants:**
```python
SAMPLE_RATE = 22050          # Audio resampling rate
DURATION = 3                  # Fixed analysis window (seconds)
SAMPLES_PER_TRACK = 66150     # 22050 * 3
N_MFCC = 40                   # Number of MFCC coefficients
```

**TensorFlow CPU Fix:**
```python
tf.config.set_visible_devices([], 'GPU')  # Force CPU for TF
```

**Preprocessing Pipeline:**
1. Load audio: `librosa.load(audio_source, sr=SAMPLE_RATE)`
2. Truncate: Clip to `SAMPLES_PER_TRACK` if longer
3. Pad: Center-pad with zeros if shorter
4. Extract MFCC: `librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)`
5. Transpose: `mfcc.T` for time-series format
6. Reshape: Add batch dimension `[1, time, features]`

Output shape: `(1, 259, 40)` where 259 = 22050*3/256

### 5.2 Emotion Detection (`emotion_model.py`)

**Class:** `EmotionPredictor`

**Emotion Mapping:**
```python
EMOTIONS = {
    0: 'angry',
    1: 'disgust', 
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad'
}
```

**Methods:**
- `__init__(model_path)` - Loads Keras model with `compile=False`
- `predict(audio_input)` - Returns `(emotion, confidence, prediction_time)`

### 5.3 Language Detection (`language_model.py`)

**Class:** `MultiLanguagePredictor`

**Architecture:** Ensemble of 3 binary classifiers:
- `hindi_vs_nonhindi_detection_model.keras`
- `english_vs_nonenglish_detection_model.keras`  
- `bengali_vs_nonbengali_detection_model.keras`

**Threshold:** 0.5 (positive if probability >= 0.5)

**Parallel Loading:**
```python
with ThreadPoolExecutor(max_workers=3) as executor:
    # Load all 3 models concurrently
```

### 5.4 Audio Transcription (`audio_text_model.py`)

**Class:** `AudioTextPredictor`

**Configuration:**
```python
WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="int8_float16"  # For 4GB VRAM
)
```

**Transcription Features:**
- `beam_size=2` - Balance speed vs accuracy
- `vad_filter=True` - Skip silent segments

**Output:** `(text, time, language, probability)`

### 5.5 API Routes (`main_api.py`)

**Key Manager:**
```python
class KeyManager:
    def get_current_key(self) -> str
    def rotate_key(self)
```

**Routes:**

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Status & GPU info |
| `/predict_emotion` | POST | Emotion analysis |
| `/M_predict_language` | POST | Language detection |
| `/predict_text` | POST | Speech transcription |
| `/refine-text` | POST | Text refinement (Gemini) |
| `/docs` | GET | Swagger UI (protected) |
| `/openapi.json` | GET | OpenAPI schema (protected) |

---

## 6. Frontend Documentation

### 6.1 Flask Application (`app.py`)

**Configuration:**
```python
USERS_FILE = 'users.json'
BASE_URL = os.getenv("API", "http://127.0.0.1:8000")
```

**Authentication:**
- Password hashing: `werkzeug.security.generate_password_hash`
- Session-based: `flask.session`

**Routes:**

| Route | Method | Description |
|-------|--------|-------------|
| `/login` | GET/POST | User authentication |
| `/register` | GET/POST | User registration |
| `/logout` | GET | Clear session |
| `/` | GET | Dashboard (protected) |
| `/audio_only` | POST | Emotion + Language API |
| `/transcribe_only` | POST | Transcription API |

**Concurrent API Calls:**
```python
with concurrent.futures.ThreadPoolExecutor() as executor:
    lang_fut = executor.submit(call_api, LANG_API_URL, ...)
    emo_fut = executor.submit(call_api, EMOTION_API_URL, ...)
```

### 6.2 Templates

- **login.html** - Authentication form
- **register.html** - Registration form with password confirmation
- **index.html** - Main dashboard for audio analysis

---

## 7. Model Architecture

### 7.1 Emotion Model

**Input:** `(batch_size, 259, 40, 1)` - MFCC spectrogram

**Architecture:**
```
Conv2D(32, 3x3, relu) -> MaxPool(2x2)
Conv2D(64, 3x3, relu) -> MaxPool(2x2)
Conv2D(128, 3x3, relu) -> MaxPool(2x2)
Flatten
Dense(128, relu)
Dropout(0.5)
Dense(6, softmax)
```

### 7.2 Language Models

**3 Binary Classifiers:**

Each architecture:
```
Conv2D(32, 3x3, relu) -> MaxPool(2x2)
Conv2D(64, 3x3, relu) -> MaxPool(2x2)
Flatten
Dense(64, relu)
Dropout(0.3)
Dense(1, sigmoid)
```

**Training:**
- Hindi: Hindi (1) vs Others (0)
- English: English (1) vs Others (0)
- Bengali: Bengali (1) vs Others (0)

### 7.3 Whisper Model

**Model:** `large-v3-turbo`
- 1.5B parameters
- Multilingual (99 languages)
- Quantization: int8_float16 (4GB VRAM compatible)

---

## 8. API Specifications

### 8.1 Endpoints

#### Health Check
```http
GET /
```

**Response:**
```json
{
    "status": "active",
    "docs_url": "/docs",
    "gpu_status": "GPU running Whisper, CPU running TensorFlow",
    "Build by": "Aditya Choudhary"
}
```

#### Emotion Prediction
```http
POST /predict_emotion
Content-Type: multipart/form-data

audio_file: <File>
```

**Response:**
```json
{
    "emotion": "happy",
    "confidence": "92.45%",
    "prediction_time": "0.3421 seconds"
}
```

#### Language Prediction
```http
POST /M_predict_language
Content-Type: multipart/form-data

audio_file: <File>
```

**Response:**
```json
{
    "predicted_language": "English",
    "prediction_time": "0.2847 seconds"
}
```

#### Speech Transcription
```http
POST /predict_text
Content-Type: multipart/form-data

audio_file: <File>
```

**Response:**
```json
{
    "transcribed_text": "Hello world",
    "detected_language": "en",
    "language_probability": "98.32%",
    "prediction_time": "1.2456 seconds"
}
```

#### Text Refinement
```http
POST /refine-text
Content-Type: application/json

{"text": "input text"}
```

**Response:**
```json
{"processed_text": "refined output"}
```

### 8.2 cURL Examples

```bash
# Emotion detection
curl -X POST "http://localhost:8000/predict_emotion" \
  -F "audio_file=@sample.wav"

# Language detection
curl -X POST "http://localhost:8000/M_predict_language" \
  -F "audio_file=@sample.wav"

# Transcription
curl -X POST "http://localhost:8000/predict_text" \
  -F "audio_file=@sample.wav"

# Text refinement
curl -X POST "http://localhost:8000/refine-text" \
  -H "Content-Type: application/json" \
  -d '{"text": "your text here"}'
```

---

## 9. Environment Setup

### 9.1 Complete .env Example

```env
# ==========================================
# BACKEND CONFIGURATION
# ==========================================

# Model Paths
EMOTION_MODEL_PATH="backend/models/emotion_model.keras"
HINDI_MODEL_PATH="backend/models/hindi_vs_nonhindi_detection_model.keras"
ENGLISH_MODEL_PATH="backend/models/english_vs_nonenglish_detection_model.keras"
BENGALI_MODEL_PATH="backend/models/bengali_vs_nonbengali_detection_model.keras"

# Google Gemini API Keys
GEMINI_API_KEY_1=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
GEMINI_API_KEY_2=AIzaSyYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
GEMINI_API_KEY_3=
GEMINI_API_KEY_4=
GEMINI_API_KEY_5=
GEMINI_API_KEY_6=

# API Docs Security
DOCS_USERNAME=admin
DOCS_PASSWORD=your_secure_password

# ==========================================
# FRONTEND CONFIGURATION
# ==========================================

API=http://127.0.0.1:8000
SECRET_KEY=your_random_secret_key_minimum_16_chars
```

### 9.2 Windows Setup

```powershell
# 1. Install Python 3.10+, FFmpeg, CUDA 12.1

# 2. Setup project
cd d:\audio_api_backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Create .env file

# 4. Run
.\run.ps1
```

### 9.3 Linux/Mac Setup

```bash
# Install dependencies
sudo apt install python3.10 python3-pip ffmpeg

# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run (two terminals)
# Terminal 1:
cd backend && uvicorn main_api:app --reload --port 8000

# Terminal 2:
cd frontend && python app.py
```

---

## 10. Deployment Guide

### 10.1 Local Development

Use `run.ps1` (Windows):
- Starts backend on port 8000
- Opens new terminal for frontend on port 5000

### 10.2 Cloudflare Tunnel (Public URL)

```powershell
# 1. Start backend locally

# 2. Create tunnel for frontend
.\cloudflared-windows-amd64.exe tunnel --url http://127.0.0.1:5000

# 3. Share the generated URL
```

Update `API` in `.env` to the tunnel URL.

### 10.3 Production Checklist

**Security:**
- [ ] Change default `DOCS_USERNAME` and `DOCS_PASSWORD`
- [ ] Generate strong `SECRET_KEY`
- [ ] Use HTTPS in production
- [ ] Add rate limiting
- [ ] Validate file uploads

**Performance:**
- [ ] Use gunicorn instead of Flask dev server
- [ ] Add Redis for session storage
- [ ] Implement request caching

---

## 11. Troubleshooting

| Issue | Solution |
|-------|----------|
| **FFmpeg not found** | Install FFmpeg and add to PATH |
| **CUDA out of memory** | Change `compute_type` from `int8_float16` to `int8` |
| **TensorFlow using GPU** | Verify `tf.config.set_visible_devices([], 'GPU')` in api_config.py |
| **Import errors** | Reinstall requirements: `pip install -r requirements.txt` |
| **Gemini API failures** | Verify keys at makersuite.google.com |
| **Frontend can't connect** | Check backend running on port 8000 |
| **Slow processing** | Verify GPU active: `nvidia-smi` |
| **PowerShell permission** | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |

### Common Error Solutions

**GPU OOM Error:**
```python
# In audio_text_model.py, change:
compute_type="int8"  # Instead of int8_float16
```

**FFmpeg Missing:**
```bash
# Windows: Download from ffmpeg.org, add to PATH
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg
```

**CUDA Version Mismatch:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## License

Built by **Aditya Choudhary**

---

*For quick reference, see README.md*
