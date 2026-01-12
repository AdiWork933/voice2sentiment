import os
import io
import shutil
import logging
import secrets
from typing import List, Optional

# --- Third Party Imports ---
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydub import AudioSegment
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from google.generativeai.types import GenerationConfig

# --- Local Imports ---
# Ensure these exist in your project structure
import api_config
from language_model import MultiLanguagePredictor

# ======================================================================
# ------------------------- CONFIGURATION ------------------------------
# ======================================================================

load_dotenv()

# --- Security Config ---
DOCS_USERNAME = os.getenv("DOCS_USERNAME", "admin")
DOCS_PASSWORD = os.getenv("DOCS_PASSWORD", "secret123")

# Gemini Configuration
VOICE_MODEL_NAME = "gemini-2.5-flash"
TEXT_MODEL_NAME = "gemini-2.0-flash"

API_KEYS = [
    os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 6)
    if os.getenv(f"GEMINI_API_KEY_{i}")
]

if not API_KEYS:
    raise RuntimeError("No API keys found in .env file. Please set GEMINI_API_KEY_1 through 5.")

MULTI_MODEL_PATHS = {
    'hindi': "models/hindi_vs_nonhindi_detection_model.keras",
    'english': "models/english_vs_nonenglish_detection_model.keras",
    'bengali': "models/bengali_vs_nonbengali_detection_model.keras"
}

logging.basicConfig(level=logging.INFO)

# ======================================================================
# ------------------------- APP SETUP ----------------------------------
# ======================================================================

app = FastAPI(
    title="AI Audio & Text Analysis API",
    # NOTICE: The text below is pushed all the way to the left
    description="""Unified API for Emotion Detection, Language ID, and Gemini Voice/Text Services.

**Source Code:** [View on GitHub](https://github.com/AdiWork933/voice2sentiment)
""",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

security = HTTPBasic()

# ======================================================================
# ------------------------- SECURITY LOGIC -----------------------------
# ======================================================================

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, DOCS_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, DOCS_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# ======================================================================
# ------------------------- SECURED DOCS ROUTES ------------------------
# ======================================================================

@app.get("/docs", include_in_schema=False)
async def get_swagger_documentation(username: str = Depends(get_current_username)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")

@app.get("/redoc", include_in_schema=False)
async def get_redoc_documentation(username: str = Depends(get_current_username)):
    return get_redoc_html(openapi_url="/openapi.json", title="docs")

@app.get("/openapi.json", include_in_schema=False)
async def openapi(username: str = Depends(get_current_username)):
    return app.openapi()

# ======================================================================
# ------------------------- GLOBAL INSTANCES ---------------------------
# ======================================================================

multi_lang_predictor = None

class KeyManager:
    """Manages rotation of Gemini API keys."""
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.current_index = 0

    def get_current_key(self) -> str:
        return self.keys[self.current_index]

    def rotate_key(self):
        prev_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.keys)
        logging.info(f"Rotating API Key: Switched from Key #{prev_index+1} to Key #{self.current_index+1}")

key_manager = KeyManager(API_KEYS)

class TextRequest(BaseModel):
    text: str

# ======================================================================
# ----------------------- HELPER FUNCTIONS -----------------------------
# ======================================================================

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        logging.critical("'ffmpeg' not found. Pydub will fail.")
        return False
    return True

def convert_audio_to_wav_bytes(audio_bytes: bytes, filename: str):
    if not check_ffmpeg():
        return None
    try:
        logging.info(f"Converting file: {filename}")
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav", codec="pcm_s16le")
        wav_buffer.seek(0)
        return wav_buffer
    except Exception as e:
        logging.error(f"Conversion Error ({filename}): {e}")
        return None

def get_audio_buffer(audio_bytes: bytes, filename: str):
    if filename.lower().endswith(".wav"):
        return io.BytesIO(audio_bytes)
    else:
        return convert_audio_to_wav_bytes(audio_bytes, filename)

def generate_content_with_retry(model_name: str, contents: list, config: types.GenerateContentConfig = None):
    """
    Tries one API key at a time. 
    If an API fails (for any reason), it shifts to the next one.
    If all APIs fail, it returns None (which is handled by the caller).
    """
    attempts = 0
    max_attempts = len(key_manager.keys)

    while attempts < max_attempts:
        current_key = key_manager.get_current_key()
        client = genai.Client(api_key=current_key)
        
        try:
            # Attempt to generate content
            response = client.models.generate_content(
                model=model_name, 
                contents=contents, 
                config=config
            )
            return response.text

        except Exception as e:
            # On ANY failure, log it and rotate to the next key
            logging.warning(f"API Key #{key_manager.current_index+1} failed: {e}. shifting to next key...")
            key_manager.rotate_key()
            attempts += 1

    # If the loop finishes, all keys have failed
    logging.error("All API keys failed to generate content.")
    return None

# ======================================================================
# --------------------------- STARTUP ----------------------------------
# ======================================================================

@app.on_event("startup")
async def startup_event():
    global multi_lang_predictor
    if hasattr(api_config, 'load_models'):
        api_config.load_models()
    logging.info("Initializing Multi-Language Predictor...")
    try:
        multi_lang_predictor = MultiLanguagePredictor(MULTI_MODEL_PATHS)
    except Exception as e:
        logging.error(f"Failed to load MultiLanguagePredictor: {e}")

# ======================================================================
# -------------------------- ROUTES ------------------------------------
# ======================================================================

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Unified Audio & Text API is running.",
        "docs_url": "/docs (Protected)",
        "team": "Voice2Sentimental"
    }

@app.get("/health")
def health_check():
    return {"status": "active", "active_gemini_key_index": key_manager.current_index + 1}

# --- Local Model Endpoints ---

@app.post("/predict_emotion")
async def predict_emotion(audio_file: UploadFile = File(...)):
    if not hasattr(api_config, 'emotion_model') or api_config.emotion_model is None:
        raise HTTPException(status_code=503, detail="Emotion model is not loaded.")

    audio_bytes = await audio_file.read()
    audio_buffer = get_audio_buffer(audio_bytes, audio_file.filename)

    if not audio_buffer:
        raise HTTPException(status_code=400, detail="Could not process/convert audio.")

    input_features = api_config.preprocess_audio(audio_buffer)

    if input_features is None:
        raise HTTPException(status_code=400, detail="Audio preprocessing failed.")

    try:
        prediction = api_config.emotion_model.predict(input_features, verbose=0)[0]
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_emotion = api_config.EMOTIONS.get(predicted_index, "Unknown")

        return {
            "filename": audio_file.filename,
            "predicted_emotion": predicted_emotion,
            "confidence_percent": f"{confidence * 100:.2f}%"
        }
    except Exception as e:
        logging.error(f"Emotion Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

@app.post("/predict_language")
async def predict_language_single_model(audio_file: UploadFile = File(...)):
    if not hasattr(api_config, 'language_model') or api_config.language_model is None:
        raise HTTPException(status_code=503, detail="Language model not loaded.")

    audio_bytes = await audio_file.read()
    audio_buffer = get_audio_buffer(audio_bytes, audio_file.filename)

    if not audio_buffer:
        raise HTTPException(status_code=400, detail="Could not process audio.")

    input_features = api_config.preprocess_audio(audio_buffer)
    
    if input_features is None:
        raise HTTPException(status_code=400, detail="Audio preprocessing failed.")

    try:
        probabilities = api_config.language_model.predict(input_features, verbose=0)[0]
        predicted_id = np.argmax(probabilities)
        predicted_language = api_config.ID_TO_LANGUAGE.get(predicted_id, "Unknown")
        confidence = probabilities[predicted_id] * 100

        full_probabilities = {
            name: f"{probabilities[id_val] * 100:.2f}%"
            for id_val, name in api_config.ID_TO_LANGUAGE.items()
        }

        return {
            "filename": audio_file.filename,
            "predicted_language": predicted_language,
            "confidence_percent": f"{confidence:.2f}%",
            "full_probabilities": full_probabilities
        }
    except Exception as e:
        logging.error(f"Language Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

@app.post("/M_predict_language")
async def predict_language_multi_binary(audio_file: UploadFile = File(...)):
    if multi_lang_predictor is None:
        raise HTTPException(status_code=503, detail="Multi-Language models not loaded.")

    audio_bytes = await audio_file.read()
    audio_buffer = get_audio_buffer(audio_bytes, audio_file.filename)

    if not audio_buffer:
        raise HTTPException(status_code=400, detail="Could not process audio.")

    try:
        final_result = multi_lang_predictor.predict(audio_buffer)
        return {"predicted_language": final_result}
    except Exception as e:
        logging.error(f"Multi-Model Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Language prediction error.")

# --- Gemini AI Endpoints ---

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Voice to Text using Gemini 2.5 Flash with Key Rotation."""
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file.")

    try:
        audio_content = await audio_file.read()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read file.")

    prompt = "Transcribe this audio file strictly. Output only the text, no conversational filler."
    
    transcription = generate_content_with_retry(
        model_name=VOICE_MODEL_NAME,
        contents=[
            prompt,
            types.Part.from_bytes(data=audio_content, mime_type=audio_file.content_type)
        ],
        config=types.GenerateContentConfig(temperature=0.0)
    )
    
    # Check if all APIs failed
    if transcription is None:
        transcription = "failed to generate"

    return {
        "filename": audio_file.filename,
        "model": VOICE_MODEL_NAME,
        "transcription": transcription
    }

# @app.post("/refine-text")
# async def refine_text(request: TextRequest):
#     """Gemini 2.0 Text Refinement / Study Helper."""
#     system_instruction = """
#     You are an intelligent text processor. Analyze the user's input text.
    
#     CONDITION 1: IF the text is educational, academic, or study-related:
#     - Format it neatly (use headers, bullet points).
#     - Add a clear "Description" section summarizing the topic.
#     - Expand slightly to ensure concepts are clear.
    
#     CONDITION 2: IF the text is general (casual, email, conversational, etc.):
#     - Strictly refine the grammar, spelling, and punctuation.
#     - Improve clarity but DO NOT change the meaning or add new info.
#     - Return only the refined text.
#     """

#     response_text = generate_content_with_retry(
#         model_name=TEXT_MODEL_NAME,
#         contents=[system_instruction, request.text],
#         config=types.GenerateContentConfig(temperature=0.3)
#     )

#     if response_text is None:
#         response_text = "failed to generate"

#     return {
#         "model": TEXT_MODEL_NAME,
#         "processed_text": response_text
#     }


# --- CONFIGURATION ---
# --- CONFIGURATION ---
TEXT_MODEL_NAME = "gemini-1.5-flash" 

# --- DATA MODELS ---
class TextRequest(BaseModel):
    text: str

# --- THE ROUTE ---
@app.post("/refine-text")
async def refine_text(request: TextRequest):
    """Gemini Text Refinement / Study Helper using the NEW google.genai SDK."""
    
    # 1. Define the System Instruction
    system_instruction = """
    You are an intelligent text processor. Analyze the user's input text.
    
    CONDITION 1: IF the text is educational, academic, or study-related:
    - Format it neatly (use headers, bullet points).
    - Add a clear "Description" section summarizing the topic.
    - Expand slightly to ensure concepts are clear.
    
    CONDITION 2: IF the text is general (casual, email, conversational, etc.):
    - Strictly refine the grammar, spelling, and punctuation.
    - Improve clarity but DO NOT change the meaning or add new info.
    - Return only the refined text.
    """

    try:
        # 2. Initialize the Client (New SDK Style)
        # We use os.getenv directly here to make it independent as requested.
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
             # Fallback to one of the numbered keys if the main one isn't set
            api_key = os.getenv("GEMINI_API_KEY_1")
            
        client = genai.Client(api_key=api_key)

        # 3. Call the API asynchronously (New SDK Style uses .aio)
        response = await client.aio.models.generate_content(
            model=TEXT_MODEL_NAME,
            contents=request.text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )
        )

        # 4. Extract text
        if response.text:
            response_text = response.text
        else:
            response_text = "No content generated."

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")

    return {
        "model": TEXT_MODEL_NAME,
        "processed_text": response_text

    }
