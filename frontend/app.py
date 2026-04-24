import os
import json
import requests
import concurrent.futures
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super_secret_production_key_123!")

# --- Configuration ---
USERS_FILE = 'users.json'
BASE_URL = os.getenv("API", "https://folks-arg-cached-feedback.trycloudflare.com ")

# API Endpoints
LANG_API_URL = f"{BASE_URL}/M_predict_language"
EMOTION_API_URL = f"{BASE_URL}/predict_emotion"
TEXT_API_URL = f"{BASE_URL}/predict_text"

# --- User Management ---
def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# --- Login Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes: Auth ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        
        if username in users and check_password_hash(users[username]['password'], password):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            error = 'Invalid Credentials. Please try again.'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            error = 'Passwords do not match.'
        else:
            users = load_users()
            if username in users:
                error = 'Username already exists.'
            else:
                users[username] = {'password': generate_password_hash(password)}
                save_users(users)
                success = 'Registration successful! You can now log in.'
                
    return render_template('register.html', error=error, success=success)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session.get('username', 'User'))

# --- Helper Functions ---
def call_api(url, audio_bytes, filename):
    try:
        files = [('audio_file', (filename, audio_bytes, 'audio/wav'))]
        response = requests.post(url, files=files, timeout=120) 
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# --- ROUTE: Audio Only (Emotion & Lang) ---
@app.route('/audio_only', methods=['POST'])
@login_required
def audio_only():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    filename = audio_file.filename
    
    try:
        audio_file.seek(0)
        audio_bytes = audio_file.read()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            lang_fut = executor.submit(call_api, LANG_API_URL, audio_bytes, filename)
            emo_fut = executor.submit(call_api, EMOTION_API_URL, audio_bytes, filename)
            
            lang_data = lang_fut.result()
            emo_data = emo_fut.result()

        return jsonify({
            "language_data": lang_data,
            "emotion_data": emo_data
        })

    except Exception as e:
        return jsonify({"error": "Failed to process audio."}), 500

# --- ROUTE: Transcribe Only ---
@app.route('/transcribe_only', methods=['POST'])
@login_required
def transcribe_only():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    filename = audio_file.filename
    
    try:
        audio_file.seek(0)
        audio_bytes = audio_file.read()
        trans_data = call_api(TEXT_API_URL, audio_bytes, filename)
        
        return jsonify({
            "transcription_data": trans_data
        })

    except Exception as e:
        return jsonify({"error": "Failed to transcribe file."}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000)