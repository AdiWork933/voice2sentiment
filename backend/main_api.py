import os
import io
import shutil
import logging
import secrets
from typing import List

# --- Third Party Imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydub import AudioSegment
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types

# --- Local Imports ---
import api_config as api_config
from emotion_model import EmotionPredictor
from language_model import MultiLanguagePredictor

# ======================================================================
# ------------------------- CONFIGURATION ------------------------------
# ======================================================================

load_dotenv()

# --- Security Config ---
DOCS_USERNAME = os.getenv("DOCS_USERNAME", "admin")
DOCS_PASSWORD = os.getenv("DOCS_PASSWORD", "secret123")

# Gemini Configuration
VOICE_MODEL_NAME = "gemini-2.0-flash" 
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
# ------------------------- APP SETUP ----------------------------------
# ======================================================================

app = FastAPI(
    title="AI Audio & Text Analysis API",
    description="""Unified API for Emotion Detection, Language ID, and Gemini Voice/Text Services.
**Source Code:** [View on GitHub](https://github.com/AdiWork933/voice2sentiment)""",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)
security = HTTPBasic()

# ======================================================================
# ------------------------- GLOBAL INSTANCES ---------------------------
# ======================================================================

multi_lang_predictor = None
emotion_predictor = None

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

@app.on_event("startup")
async def startup_event():
    global multi_lang_predictor, emotion_predictor
    
    # Load Emotion Model
    emotion_path = os.getenv("EMOTION_MODEL_PATH")
    emotion_predictor = EmotionPredictor(emotion_path)
        
    # Load Multi-Language Models
    logging.info("Initializing Multi-Language Predictor...")
    try:
        multi_lang_predictor = MultiLanguagePredictor(MULTI_MODEL_PATHS)
    except Exception as e:
        logging.error(f"Failed to load MultiLanguagePredictor: {e}")

# ======================================================================
# ----------------------- HELPER FUNCTIONS -----------------------------
# ======================================================================

def get_audio_buffer(audio_bytes: bytes, filename: str):
    """Converts audio to WAV buffer using Pydub/FFmpeg if needed."""
    if filename.lower().endswith(".wav"):
        return io.BytesIO(audio_bytes)
    
    if shutil.which("ffmpeg") is None:
        logging.error("FFmpeg not found. Cannot convert audio.")
        return None
        
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav", codec="pcm_s16le")
        wav_buffer.seek(0)
        return wav_buffer
    except Exception as e:
        logging.error(f"Conversion Error: {e}")
        return None

# ======================================================================
# -------------------------- ROUTES ------------------------------------
# ======================================================================

@app.get("/")
def read_root():
    return {
        "status": "active",
        "docs_url": "/docs",
        "gpu_status": "Qwen Reserved (Audio using CPU)",
        "Build by": "Aditya Choudhary"
    }

# --- Local Model Endpoints ---

@app.post("/predict_emotion")
async def predict_emotion(audio_file: UploadFile = File(...)):
    if emotion_predictor is None or emotion_predictor.model is None:
        raise HTTPException(status_code=503, detail="Emotion model not loaded.")

    audio_bytes = await audio_file.read()
    audio_buffer = get_audio_buffer(audio_bytes, audio_file.filename)
    
    if not audio_buffer:
        raise HTTPException(status_code=400, detail="Invalid audio format.")

    emotion, confidence, pred_time = emotion_predictor.predict(audio_buffer)
    
    if emotion is None:
        raise HTTPException(status_code=400, detail="Processing failed.")

    return {
        "emotion": emotion,
        "confidence": f"{confidence * 100:.2f}%",
        "prediction_time": f"{pred_time} seconds"
    }

@app.post("/M_predict_language")
async def predict_language_multi(audio_file: UploadFile = File(...)):
    if not multi_lang_predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready.")

    audio_bytes = await audio_file.read()
    audio_buffer = get_audio_buffer(audio_bytes, audio_file.filename)
    
    if not audio_buffer:
        raise HTTPException(status_code=400, detail="Invalid audio.")

    result, pred_time = multi_lang_predictor.predict(audio_buffer)
    
    return {
        "predicted_language": result,
        "prediction_time": f"{pred_time} seconds"
    }

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
