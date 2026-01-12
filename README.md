# Voice2Sentimental: Unified Audio & Text Analysis Platform

**Voice2Sentimental** is a full-stack AI application designed to analyze audio files for emotional tone, language identity, and content. It features a dual-architecture system: a high-performance **FastAPI Backend** for machine learning inference and a user-friendly **Flask Frontend** for file management and reporting.

---

## 🏗️ Architecture Overview

The system is split into two distinct services:

1.  **Backend API (`main_api.py`)**:
    * Built with **FastAPI** for high speed and asynchronous processing.
    * Hosts TensorFlow/Keras models for **Emotion Detection** and **Language Identification**.
    * Integrates with **Google Gemini** (via `google-genai`) for advanced transcription and text refinement.
    * Implements API key rotation for reliability.

2.  **Frontend Client (`app.py`)**:
    * Built with **Flask** to serve a web-based UI.
    * Handles user authentication and file uploads.
    * **Smart Chunking:** Automatically splits large audio files (>60s) into 1-minute segments to generate timeline-based emotion reports.
    * Aggregates data from the API into visual statistics and transcript logs.

---

## 🚀 Key Features

* **Emotion Detection:** Classifies audio into 6 categories (Angry, Disgust, Fear, Happy, Neutral, Sad) with confidence scores.
* **Multi-Language Detection:** Uses an ensemble of binary models to distinguish between Hindi, English, and Bengali.
* **AI Transcription:** Utilizes Google Gemini 2.5 Flash for high-accuracy speech-to-text conversion.
* **Text Refinement:** Post-processes transcripts to correct grammar or format text for educational study guides.
* **Secure API:** Backend documentation (`/docs`) is protected via HTTP Basic Auth.

---

## 🛠️ Prerequisites

* **Python 3.10+**
* **FFmpeg:** Required by `pydub` for audio processing (converting/slicing files).
    * *Windows:* Download FFmpeg and add the `/bin` folder to your System PATH.
    * *Linux/Mac:* `sudo apt install ffmpeg`
* **CUDA (Optional):** Recommended for GPU acceleration if running local PyTorch/TensorFlow models.

---

## 📦 Installation & Setup

### 1. Backend Setup (API)

### 🔧 Setup For Window OS

-   Move Directory : `cd backend`
-   Creating virtual environment: `python -m venv venv`
-   set Exicute Access(if window denied execution) : `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`
-   Activate virtual environment : `.\venv\Scripts\Activate`
-   Installing Required Librady : `pip install -r requirements.txt`
-   Run backend server : `uvicorn main_api:app --reload --port 8000` or directly run `./run.ps1`
-   To stope streamlit Server : `ctrl + c (for Stop)`
-   To Deactivate virtual environment : `deactivate ( into terminal)`


### 🔧 Setup For ubuntu OS
  
-   Move Directory : `cd backend`
-   Creating virtual environment: `python -m venv venv`
-   Activate virtual environment : `source .venv /bin/activate`
-   Installing Required Librady : `pip install -r requirements.txt`
-   Run backend server : `uvicorn main_api:app --reload --port 8000`
-   To stope streamlit Server : `ctrl + c (for Stop)`
-   To Deactivate virtual environment : `deactivate ( into terminal)`

### 2. Frontend Setup 

Note : If your running both `Backend & Frontend` into same virtual environment no need to create again   `python -m venv venv`
        Just in another terminal install requirements and setuo `.env` and run.

### 🔧 Setup For Window OS

-   Move Directory : `cd frontend`
-   Creating virtual environment(if needed): `python -m venv venv`
-   set Exicute Access(if window denied execution(if needed)) : `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`
-   Activate virtual environment(if needed) : `.\venv\Scripts\Activate`
-   Installing Required Librady : `pip install -r requirements.txt`
-   Run backend server : `python app.py`
-   To stope streamlit Server : `ctrl + c (for Stop)`
-   To Deactivate virtual environment : `deactivate ( into terminal)`


### 🔧 Setup For ubuntu OS
  
-   Move Directory : `cd frontend`
-   Creating virtual environment(if needed): `python -m venv venv`
-   Activate virtual environment(if needed): `source .venv /bin/activate`
-   Installing Required Librady(if needed) : `pip install -r requirements.txt`
-   Run backend server : `python app.py`
-   To stope streamlit Server : `ctrl + c (for Stop)`
-   To Deactivate virtual environment : `deactivate ( into terminal)`

---

## Configure Environment (.env): Create a .env file in the backend folder:

### 1. For Backend

```bash
# Model Paths
EMOTION_MODEL_PATH="models/emotion_model.keras" 
LANGUAGE_MODEL_PATH="models/language_detection_model.keras"
HINDI_MODEL_PATH="models/hindi_vs_nonhindi_detection_model.keras"
ENGLISH_MODEL_PATH="models/english_vs_nonenglish_detection_model.keras"
BENGALI_MODEL_PATH="models/bengali_vs_nonbengali_detection_model.keras"
# Google Gemini Keys (Supports rotation)
GEMINI_API_KEY_1 = your_key_here(important)
GEMINI_API_KEY_2 = your_key_here
GEMINI_API_KEY_3 = your_key_here
GEMINI_API_KEY_4 = your_key_here
GEMINI_API_KEY_5 = your_key_here
GEMINI_API_KEY_6 = your_key_here(important)
# API Documentation Security
DOCS_USERNAME=admin
DOCS_PASSWORD=admin123
```

### 2. For Frontend

```bash
# Connection to Backend
API = http://127.0.0.1:8000 (your Backend root API)

# Web App Security
SECRET_KEY = long_random_string
ADMIN_USERNAME = admin
ADMIN_PASSWORD = password
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict_emotion` | Analyze audio for emotional tone. |
| `POST` | `/M_predict_language` | Detect language (Hindi/English/Bengali). |
| `POST` | `/transcribe` | Convert speech to text using Gemini. |
| `POST` | `/refine-text` | Improve grammar or format text as study notes. |
| `GET` | `/health` | Check API status and active key index. |

---

## 📂 Project Structure

```text
├── backend/
│   ├──models/
│   ├── main_api.py         # FastAPI Entry Point
│   ├── api_config.py       # Model Loaders & Audio Preprocessing
│   ├── language_model.py   # Multi-Model Language Logic
│   └── requirements.txt
└── frontend/
    ├── app.py              # Flask Client Application
    ├── templates/          # HTML Views (Login, Dashboard)
    └── requirements.txt
```
---

## architecture of DNN

<img width="2816" height="1536" alt="Gemini_Generated_Image_sphjmtsphjmtsphj" src="https://github.com/user-attachments/assets/62a5ebde-cb17-4e5e-a50f-04a8c8f91e38" />

## Fast API docs 

<img width="1811" height="585" alt="image" src="https://github.com/user-attachments/assets/7617545a-6c2e-49d9-9535-380a04494337" />
