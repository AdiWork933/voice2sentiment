# Voice2Sentimental: Unified Audio & Text Analysis Platform

A full-stack AI application that analyzes audio files for emotional tone, language identity, and content transcription. Built with a dual-architecture: **FastAPI Backend** for ML inference and **Flask Frontend** for web interface.

---

## 🏗️ Architecture Overview

### Backend (`backend/`)
- **FastAPI** for high-performance async processing
- **TensorFlow/Keras** models for Emotion Detection & Language Identification
- **Faster-Whisper** (GPU) for Speech-to-Text transcription
- **Google Gemini API** for text refinement and grammar correction
- API key rotation system for reliability

### Frontend (`frontend/`)
- **Flask** web framework with Jinja2 templates
- User authentication system (login/register)
- Concurrent API calls for emotion + language detection
- Responsive dashboard for audio analysis

---

## ✨ Key Features

| Feature | Description | Model/Technology |
|---------|-------------|----------------|
| **Emotion Detection** | 6-class classification (Angry, Disgust, Fear, Happy, Neutral, Sad) | TensorFlow CNN on CPU |
| **Language Detection** | Identifies Hindi, English, or Bengali using binary ensemble | 3 Parallel TF Models |
| **Speech Transcription** | Converts audio to text with language detection | Faster-Whisper on GPU |
| **Text Refinement** | Grammar correction & study guide formatting | Google Gemini 1.5 Flash |
| **Secure Docs** | Password-protected API documentation | HTTP Basic Auth |

---

## � Prerequisites

- **Python 3.10+**
- **FFmpeg** (required for audio conversion):
  - Windows: Add to System PATH
  - Linux/Mac: `sudo apt install ffmpeg`
- **NVIDIA GPU** (optional): GTX 1650+ recommended for Whisper transcription
- **CUDA 12.1** (if using GPU)

---

## � Quick Start

### Clone & Setup

```bash
# Navigate to project
cd audio_api_backend

# Create virtual environment
python -m venv .venv

# Windows: Activate
.venv\Scripts\activate

# Linux/Mac: Activate
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run with Single Command (Recommended)

```powershell
# Windows PowerShell
./run.ps1
```

This script:
1. Starts FastAPI backend on port 8000
2. Opens new terminal for Flask frontend on port 5000

### Manual Start

```bash
# Terminal 1 - Backend
cd backend
uvicorn main_api:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
python app.py
```

---

## ⚙️ Environment Configuration

Create `.env` in project root with both backend and frontend configs:

### Backend Variables

```env
# Model Paths (relative to backend/)
EMOTION_MODEL_PATH="backend/models/emotion_model.keras"
HINDI_MODEL_PATH="backend/models/hindi_vs_nonhindi_detection_model.keras"
ENGLISH_MODEL_PATH="backend/models/english_vs_nonenglish_detection_model.keras"
BENGALI_MODEL_PATH="backend/models/bengali_vs_nonbengali_detection_model.keras"

# Google Gemini API Keys (supports rotation)
GEMINI_API_KEY_1=your_key_1
GEMINI_API_KEY_2=your_key_2
# ... up to GEMINI_API_KEY_6

# API Docs Security
DOCS_USERNAME=admin
DOCS_PASSWORD=secure_password
```

### Frontend Variables

```env
# Backend Connection
API=http://127.0.0.1:8000

# Flask Security
SECRET_KEY=your_random_secret_key_here
```

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Request | Response |
|--------|----------|---------|----------|
| `GET` | `/` | - | Status, GPU info |
| `POST` | `/predict_emotion` | `audio_file` (Upload) | `{emotion, confidence, prediction_time}` |
| `POST` | `/M_predict_language` | `audio_file` (Upload) | `{predicted_language, prediction_time}` |
| `POST` | `/predict_text` | `audio_file` (Upload) | `{transcribed_text, detected_language, language_probability, prediction_time}` |
| `POST` | `/refine-text` | `{text: string}` | `{processed_text}` |
| `GET` | `/docs` | Basic Auth | Swagger UI |

### Example: Emotion Detection

```bash
curl -X POST "http://localhost:8000/predict_emotion" \
  -H "accept: application/json" \
  -F "audio_file=@sample_audio.wav"
```

**Response:**
```json
{
  "emotion": "happy",
  "confidence": "92.45%",
  "prediction_time": "0.3421 seconds"
}
```

---

## 📂 Project Structure

```
audio_api_backend/
├── backend/
│   ├── api_config.py          # Audio preprocessing, MFCC extraction, TF CPU config
│   ├── emotion_model.py       # EmotionPredictor class (6-class CNN)
│   ├── language_model.py      # MultiLanguagePredictor (3 binary models)
│   ├── audio_text_model.py    # AudioTextPredictor (Faster-Whisper GPU)
│   ├── main_api.py            # FastAPI routes, key rotation, auth
│   ├── models/                # Trained .keras models
│   └── evaluate_dataset.py    # Model evaluation utilities
│
├── frontend/
│   ├── app.py                 # Flask app, auth, API client
│   ├── templates/             # HTML templates (login, index, register)
│   └── users.json             # User credentials storage
│
├── requirements.txt           # Dependencies
├── run.ps1                    # One-click launcher (Windows)
└── .env                       # Environment variables (not in git)
```

---

## 🧠 Model Architecture

### Emotion Detection Model
- **Input:** 3-second audio clip (22050 Hz)
- **Features:** 40 MFCC coefficients
- **Architecture:** CNN with Conv2D layers
- **Output:** 6 emotion classes (Softmax)
- **Runtime:** TensorFlow on CPU

### Language Detection
- **Approach:** 3 binary classifiers (One-vs-Rest)
- **Models:** Hindi vs Non-Hindi, English vs Non-English, Bengali vs Non-Bengali
- **Ensemble Logic:** Highest probability above 0.5 threshold wins
- **Runtime:** Parallel ThreadPoolExecutor loading

### Speech Transcription
- **Model:** Faster-Whisper large-v3-turbo
- **Device:** CUDA GPU (int8_float16 for 4GB VRAM compatibility)
- **Features:** VAD filtering for silence removal
- **Output:** Text + detected language + confidence

---

## 🔒 Security Features

- **API Docs Protection:** HTTP Basic Auth for `/docs` and `/openapi.json`
- **Password Hashing:** Werkzeug security for user passwords
- **Session Management:** Flask secure sessions
- **Key Rotation:** Automatic fallback on Gemini API failures

---

## 🖼️ Screenshots

### API Documentation (Swagger UI)
![API Docs](https://github.com/user-attachments/assets/7617545a-6c2e-49d9-9535-380a04494337)

### Neural Network Architecture
![Architecture](https://github.com/user-attachments/assets/62a5ebde-cb17-4e5e-a50f-04a8c8f91e38)

---

## 🛠️ Development Notes

### GPU/CPU Workload Split
- **TensorFlow:** Restricted to CPU (avoid OOM on 4GB VRAM)
- **Whisper:** Uses CUDA GPU for transcription speed

### Audio Preprocessing Pipeline
1. Load audio with librosa (22050 Hz)
2. Pad/trim to 3 seconds
3. Extract 40 MFCC features
4. Reshape for CNN input

---

## 📄 License

Built by **Aditya Choudhary**

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| FFmpeg not found | Install and add to PATH |
| GPU out of memory | Use `int8` instead of `int8_float16` in Whisper |
| Import errors | Ensure all requirements installed |
| Permission denied (Windows) | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |
