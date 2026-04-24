import os
import site

# ======================================================================
# 🚨 WINDOWS GPU DLL FIX (MUST BE AT THE VERY TOP) 🚨
# Forces Windows to find the CUDA libraries hidden inside PyTorch
# ======================================================================
try:
    site_packages = site.getsitepackages()[0]
    torch_lib_path = os.path.join(site_packages, "torch", "lib")
    os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import io
import shutil
import logging
import secrets
import tempfile
from typing import List
from contextlib import asynccontextmanager

# --- Third Party Imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from pydub import AudioSegment
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types

# --- Local Imports ---
import api_config
from emotion_model import EmotionPredictor
from language_model import MultiLanguagePredictor
from audio_text_model import AudioTextPredictor

# ======================================================================
# ------------------------- CONFIGURATION ------------------------------
# ======================================================================

load_dotenv()

# --- Security Config ---
DOCS_USERNAME = os.getenv("DOCS_USERNAME", "admin")
DOCS_PASSWORD = os.getenv("DOCS_PASSWORD", "secret123")

# Gemini Configuration
TEXT_MODEL_NAME = "gemini-1.5-flash"

# Load API Keys safely
API_KEYS = [
    os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 6)
    if os.getenv(f"GEMINI_API_KEY_{i}")
]

if not API_KEYS and os.getenv("GEMINI_API_KEY"):
    API_KEYS.append(os.getenv("GEMINI_API_KEY"))

if not API_KEYS:
    logging.warning("⚠️ No Gemini API keys found in .env file.")

MULTI_MODEL_PATHS = {
    'hindi': os.getenv("HINDI_MODEL_PATH"),
    'english': os.getenv("ENGLISH_MODEL_PATH"),
    'bengali': os.getenv("BENGALI_MODEL_PATH")
}

logging.basicConfig(level=logging.INFO)

# ======================================================================
# ------------------------- GLOBAL INSTANCES ---------------------------
# ======================================================================

multi_lang_predictor = None
emotion_predictor = None
audio_text_predictor = None

class KeyManager:
    """Manages rotation of Gemini API keys."""
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.current_index = 0

    def get_current_key(self) -> str:
        if not self.keys: return None
        return self.keys[self.current_index]

    def rotate_key(self):
        if not self.keys: return
        prev_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.keys)
        logging.info(f"Rotating API Key: Switched from #{prev_index+1} to #{self.current_index+1}")

key_manager = KeyManager(API_KEYS)

class TextRequest(BaseModel):
    text: str

# ======================================================================
# --------------------------- STARTUP ----------------------------------
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global multi_lang_predictor, emotion_predictor, audio_text_predictor
    
    logging.info("Initializing Audio-to-Text Predictor...")
    audio_text_predictor = AudioTextPredictor()
    
    # Load Emotion Model
    emotion_path = os.getenv("EMOTION_MODEL_PATH")
    emotion_predictor = EmotionPredictor(emotion_path)
        
    # Load Multi-Language Models
    logging.info("Initializing Multi-Language Predictor...")
    try:
        multi_lang_predictor = MultiLanguagePredictor(MULTI_MODEL_PATHS)
    except Exception as e:
        logging.error(f"Failed to load MultiLanguagePredictor: {e}")
        
    yield  # The application runs here
    # Cleanup logic (if any) goes here

# ======================================================================
# ------------------------- APP SETUP ----------------------------------
# ======================================================================

app = FastAPI(
    title="AI Audio & Text Analysis API",
    description="""Unified API for Emotion Detection, Language ID, and Gemini Voice/Text Services.""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)
security = HTTPBasic()

# ======================================================================
# ----------------------- HELPER FUNCTIONS -----------------------------
# ======================================================================

def save_as_temp_wav(audio_bytes: bytes, filename: str) -> str:
    """Safely saves bytes to a temporary WAV file to prevent ML library crashes."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()

    if filename.lower().endswith(".wav"):
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        return temp_path
    
    if shutil.which("ffmpeg") is None:
        logging.error("FFmpeg not found. Cannot convert non-WAV audio.")
        os.remove(temp_path)
        return None
        
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio.export(temp_path, format="wav", codec="pcm_s16le")
        return temp_path
    except Exception as e:
        logging.error(f"Conversion Error: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

# ======================================================================
# -------------------------- ROUTES ------------------------------------
# ======================================================================

@app.get("/")
def read_root():
    return {
        "status": "active",
        "docs_url": "/docs",
        "gpu_status": "GPU running Whisper, CPU running TensorFlow",
        "Build by": "Aditya Choudhary"
    }

# --- Local Model Endpoints ---

@app.post("/predict_emotion")
async def predict_emotion(audio_file: UploadFile = File(...)):
    if emotion_predictor is None or emotion_predictor.model is None:
        raise HTTPException(status_code=503, detail="Emotion model not loaded.")

    audio_bytes = await audio_file.read()
    temp_path = save_as_temp_wav(audio_bytes, audio_file.filename)
    
    if not temp_path:
        raise HTTPException(status_code=400, detail="Invalid audio format or FFmpeg missing.")

    try:
        emotion, confidence, pred_time = emotion_predictor.predict(temp_path)
        if emotion is None:
            raise HTTPException(status_code=400, detail="Processing failed.")

        return {
            "emotion": emotion,
            "confidence": f"{confidence * 100:.2f}%",
            "prediction_time": f"{pred_time} seconds"
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/M_predict_language")
async def predict_language_multi(audio_file: UploadFile = File(...)):
    if not multi_lang_predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready.")

    audio_bytes = await audio_file.read()
    temp_path = save_as_temp_wav(audio_bytes, audio_file.filename)
    
    if not temp_path:
        raise HTTPException(status_code=400, detail="Invalid audio format or FFmpeg missing.")

    try:
        result, pred_time = multi_lang_predictor.predict(temp_path)
        return {
            "predicted_language": result,
            "prediction_time": f"{pred_time} seconds"
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict_text")
async def predict_text(audio_file: UploadFile = File(...)):
    if not audio_text_predictor or not audio_text_predictor.model:
        raise HTTPException(status_code=503, detail="Text model not loaded.")

    audio_bytes = await audio_file.read()
    temp_path = save_as_temp_wav(audio_bytes, audio_file.filename)
    
    if not temp_path:
        raise HTTPException(status_code=400, detail="Invalid audio format or FFmpeg missing.")

    try:
        text, pred_time, detected_lang, prob = audio_text_predictor.predict(temp_path)
        return {
            "transcribed_text": text,
            "detected_language": detected_lang,
            "language_probability": f"{prob * 100:.2f}%",
            "prediction_time": f"{pred_time} seconds"
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Gemini Endpoints ---

@app.post("/refine-text")
async def refine_text(request: TextRequest):
    """Refine text using Gemini with Key Rotation."""
    system_instruction = """
    Analyze the user's input text.
    - If educational: Format nicely, add headers, summarize.
    - If casual: Fix grammar/spelling only. Return refined text.
    """

    attempts = 0
    max_attempts = len(key_manager.keys) if key_manager.keys else 1
    
    while attempts < max_attempts:
        current_key = key_manager.get_current_key()
        if not current_key: break
        
        try:
            client = genai.Client(api_key=current_key)
            response = await client.aio.models.generate_content(
                model=TEXT_MODEL_NAME,
                contents=request.text,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3
                )
            )
            return {"processed_text": response.text if response.text else "No content."}

        except Exception as e:
            logging.warning(f"Key failure: {e}")
            key_manager.rotate_key()
            attempts += 1
            
    raise HTTPException(status_code=500, detail="All API keys failed.")

# --- Security/Docs Endpoints ---
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    if not (secrets.compare_digest(credentials.username, DOCS_USERNAME) and 
            secrets.compare_digest(credentials.password, DOCS_PASSWORD)):
        raise HTTPException(status_code=401, headers={"WWW-Authenticate": "Basic"})
    return credentials.username

@app.get("/docs", include_in_schema=False)
async def get_docs(username: str = Depends(get_current_username)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Docs")

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi(username: str = Depends(get_current_username)):
    return app.openapi()