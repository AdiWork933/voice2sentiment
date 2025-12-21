# import os
# import io
# import json
# import requests
# import concurrent.futures
# from functools import wraps
# from flask import Flask, render_template, request, jsonify, session, redirect, url_for
# from pydub import AudioSegment
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)
# app.secret_key = os.getenv("SECRET_KEY", "fallback_secret_key")

# # --- Configuration ---
# ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin")
# ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "password")
# BASE_URL = os.getenv("API")

# # API Endpoints
# # BASE_URL = "https://includes-wales-candle-advertisers.trycloudflare.com" 
# LANG_API_URL = f"{BASE_URL}/M_predict_language"
# EMOTION_API_URL = f"{BASE_URL}/predict_emotion"
# TRANSCRIBE_API_URL = f"{BASE_URL}/transcribe"
# REFINE_API_URL = f"{BASE_URL}/refine-text"

# API_HEADERS = {
#   'Cookie': '.Tunnels.Relay.WebForwarding.Cookies=CfDJ8Cs4yarcs6pKkdu0hlKHsZuYEbzb1zuO1s_WPmIHLxLnGTkfcOo_twS4i1LL3t3PQKn0yu7bECL0OWzIE6ILVPfP3IoqjdZXMlolibBYdgiIUKnJglbrY6ZMnXxSN1sPyps0znVt-7TYrbVdtsCTh5RPKrTti95xbrT3nleH1AXda4UfrpyJ5pj6WyCfpRsn5xOJG1B1KUn5hUWkAdfiZyMbdbg8fyY_sKUhy3NpYlj4dJ28DXd9XX8ZgM05AFVWfcl86dpNHkA59Q4cgI5haJSFfEVS8tWFqkmIeo-q5Pm166XqwurtyUUb-Y15ZQqePP0BUlOr7W_TA2AieIjM1EoTAcXna6Xd0sIx37TClZkwR658kN42x5AVxn6ju52nWiDNJeUv-z4tzCISqsbUWgjNtVePuTPCUObOvZ_V0qm_ygDLqeH1RmB67wN-Eb7mEzTyf7JsUFrpFBgKFBUDKyb8MiAIzd5eXhH3dgcqPhPg6LH5dCXkVAYQLE7G-4l9dTIme6oa8x9pMalvSkg_OnBENpoRFKj8wf7ECb_j7xWtyFscTGEGQQlBQ85WIV4iOBog3zp6ql6bym6aTaAdHvp-le_puIbyVDiiqOFQXIHoBQ3dzkIFSY5kIiAZjDgaBWNiUMXBAdhGZZKb3U71QhUgb7dgIEroC7pQav6KA1aer8GxEqAOUjUMoSnDrGdY0k1ZMo8_jUKqf4GYzVEomyjN7iwsk4bblmbezZHpo_470JT0X3PuLktIsGxfBgEXXGFxV0lfUQeqfZaQkTHqPEl1SnvdG6tbpqU1_eg_J6zRXteu59PiNrxo_ff-uu3EQS-GAx8mRUg35WCw0sn0K1dPttxWFh0v-1QybOnZy-oy0yk16OfUK_7jFE_zZcFULMxUY30Nd3MCYuHq4y0w5YwMdFbxgIrB9hr2hST1ZMi6'
# }

# # --- Login Decorator ---
# def login_required(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         if 'logged_in' not in session:
#             return redirect(url_for('login'))
#         return f(*args, **kwargs)
#     return decorated_function

# # --- Routes: Auth ---
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         if username == ADMIN_USER and password == ADMIN_PASS:
#             session['logged_in'] = True
#             return redirect(url_for('index'))
#         else:
#             error = 'Invalid Credentials. Please try again.'
#     return render_template('login.html', error=error)

# @app.route('/logout')
# def logout():
#     session.pop('logged_in', None)
#     return redirect(url_for('login'))

# @app.route('/')
# @login_required
# def index():
#     return render_template('index.html')

# # --- Helper Functions (Robust Error Handling) ---
# def call_api_generic(url, audio_bytes, filename, content_type):
#     try:
#         files = [('audio_file', (filename, audio_bytes, content_type))]
#         response = requests.post(url, headers=API_HEADERS, files=files, timeout=30) # Reduced timeout
#         if response.status_code == 200:
#             return response.json()
#         return {"error": f"HTTP {response.status_code}", "predicted_language": "Unknown", "predicted_emotion": "Unknown"}
#     except Exception as e:
#         print(f"API Fail {url}: {e}")
#         return {"error": str(e), "predicted_language": "Unknown", "predicted_emotion": "Unknown"}

# def call_transcribe_api(audio_bytes, filename):
#     try:
#         files = [('audio_file', (filename, audio_bytes, 'audio/wav'))]
#         response = requests.post(TRANSCRIBE_API_URL, files=files, timeout=60)
#         if response.status_code == 200:
#             return response.json()
#         return {"transcription": "", "error": f"HTTP {response.status_code}"}
#     except Exception as e:
#         print(f"Transcribe Fail: {e}")
#         return {"transcription": "", "error": str(e)}

# def call_refine_api(text):
#     if not text or len(text.strip()) == 0:
#         return {"detail": "Skipped (No text source)"}
        
#     try:
#         url = REFINE_API_URL
#         payload = json.dumps({"text": text})
#         headers = {'Content-Type': 'application/json'}
#         response = requests.post(url, headers=headers, data=payload, timeout=30)
        
#         if response.status_code == 200:
#             return response.json()
#         return {"detail": text, "error": f"Refine HTTP {response.status_code}"}
#     except Exception as e:
#         print(f"Refine Fail: {e}")
#         return {"detail": text, "error": str(e)}

# def process_chunk(chunk_audio, index, content_type):
#     buf = io.BytesIO()
#     chunk_audio.export(buf, format="wav")
#     chunk_bytes = buf.getvalue()
#     filename = f"chunk_{index}.wav"
    
#     start_time = index * 60
#     end_time = (index + 1) * 60
#     time_label = f"{start_time}m - {end_time}m"

#     # Default Results
#     lang_res = {"predicted_language": "Unknown"}
#     emo_res = {"predicted_emotion": "Unknown"}
#     trans_res = {"transcription": ""}
#     refine_res = {"detail": ""}

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # 1. First call Emotion and Language
#         future_lang = executor.submit(call_api_generic, LANG_API_URL, chunk_bytes, filename, "audio/wav")
#         future_emo = executor.submit(call_api_generic, EMOTION_API_URL, chunk_bytes, filename, "audio/wav")
        
#         # Wait for them (or just let them finish while we prep next steps, 
#         # but logic dictates we treat them as the first batch)
#         try:
#             lang_res = future_lang.result()
#         except Exception as e:
#             lang_res = {"error": str(e), "predicted_language": "Unknown"}
            
#         try:
#             emo_res = future_emo.result()
#         except Exception as e:
#             emo_res = {"error": str(e), "predicted_emotion": "Unknown"}

#         # 2. Then call Remaining (Transcription)
#         # We execute this regardless of Lang/Emo success
#         try:
#             trans_res = call_transcribe_api(chunk_bytes, filename)
#         except Exception as e:
#             trans_res = {"error": str(e), "transcription": ""}

#         # 3. Refine Logic (Depends on Transcribe Success)
#         raw_text = trans_res.get('transcription', '')
#         if raw_text:
#             try:
#                 refine_res = call_refine_api(raw_text)
#             except Exception as e:
#                 refine_res = {"detail": raw_text, "error": str(e)}
#         else:
#             refine_res = {"detail": "No transcription available to refine."}

#         return {
#             "index": index,
#             "timestamp": time_label,
#             "lang_res": lang_res,
#             "emo_res": emo_res,
#             "trans_res": trans_res,
#             "refine_res": refine_res
#         }

# @app.route('/predict', methods=['POST'])
# @login_required
# def predict():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file provided'}), 400

#     audio_file = request.files['audio']
#     filename = audio_file.filename
#     content_type = audio_file.content_type 

#     try:
#         audio = AudioSegment.from_file(audio_file)
#         duration_sec = len(audio) / 1000.0

#         # === SINGLE FILE PROCESSING (< 60s) ===
#         if duration_sec <= 60:
#             audio_file.seek(0)
#             audio_bytes = audio_file.read()
            
#             # Default containers
#             lang_data = {"predicted_language": "Unknown"}
#             emo_data = {"predicted_emotion": "Unknown"}
#             trans_data = {"transcription": ""}
#             refine_data = {"detail": ""}

#             with concurrent.futures.ThreadPoolExecutor() as executor:
#                 # Step 1: Launch Language & Emotion
#                 lang_fut = executor.submit(call_api_generic, LANG_API_URL, audio_bytes, filename, content_type)
#                 emo_fut = executor.submit(call_api_generic, EMOTION_API_URL, audio_bytes, filename, content_type)
                
#                 # Retrieve Phase 1 Results (Safely)
#                 try: lang_data = lang_fut.result()
#                 except Exception: pass
                
#                 try: emo_data = emo_fut.result()
#                 except Exception: pass

#                 # Step 2: Launch Transcription (Remaining API)
#                 try:
#                     trans_data = call_transcribe_api(audio_bytes, filename)
#                 except Exception: pass
                
#                 # Step 3: Launch Refinement (Dependent on Step 2)
#                 raw_text = trans_data.get('transcription', '')
#                 if raw_text:
#                     refine_data = call_refine_api(raw_text)
#                 else:
#                     refine_data = {"detail": "Refinement skipped (No text)"}

#                 return jsonify({
#                     "type": "single",
#                     "language_data": lang_data,
#                     "emotion_data": emo_data,
#                     "transcription_data": trans_data,
#                     "refinement_data": refine_data
#                 })

#         # === CHUNK PROCESSING (> 60s) ===
#         else:
#             chunk_length_ms = 60000 
#             chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            
#             results = []
#             # Using ThreadPool to process chunks concurrently
#             # Note: process_chunk internal logic handles the API sequence per chunk
#             with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#                 futures = [executor.submit(process_chunk, chunk, i, content_type) for i, chunk in enumerate(chunks)]
#                 for future in concurrent.futures.as_completed(futures):
#                     try:
#                         results.append(future.result())
#                     except Exception as e:
#                         print(f"Chunk processing failed: {e}")
#                         # Append a safe failure object if a chunk totally crashes
#                         results.append({
#                             "index": -1, "timestamp": "Error", 
#                             "lang_res": {"predicted_language": "Unknown"},
#                             "emo_res": {"predicted_emotion": "Unknown"}
#                         })
            
#             results.sort(key=lambda x: x['index'])

#             # --- Aggregation Logic ---
#             timeline_labels = [r['timestamp'] for r in results if r['index'] != -1]
#             emotion_timeline = [r['emo_res'].get('predicted_emotion', 'Unknown') for r in results]
            
#             lang_counts = {}
#             emotion_counts = {}
#             transcript_log = [] 

#             for r in results:
#                 if r['index'] == -1: continue # Skip failed chunks

#                 # Count Languages
#                 l = r['lang_res'].get('predicted_language', 'Unknown')
#                 lang_counts[l] = lang_counts.get(l, 0) + 1
                
#                 # Count Emotions
#                 e = r['emo_res'].get('predicted_emotion', 'Unknown')
#                 emotion_counts[e] = emotion_counts.get(e, 0) + 1
                
#                 # Collect Text
#                 transcript_log.append({
#                     "timestamp": r['timestamp'],
#                     "text": r['trans_res'].get('transcription', 'Error/Empty'),
#                     "refined": r['refine_res'].get('detail', 'Error/Empty')
#                 })

#             return jsonify({
#                 "type": "report",
#                 "duration": duration_sec,
#                 "timeline": {
#                     "labels": timeline_labels,
#                     "emotions": emotion_timeline
#                 },
#                 "stats": {
#                     "languages": lang_counts,
#                     "emotions": emotion_counts
#                 },
#                 "transcripts": transcript_log
#             })

#     except Exception as e:
#         print(f"Global Error: {e}")
#         return jsonify({"error": "Failed to process audio file."}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)



#----------------------------------------------------------------------------------------

# import os
# import io
# import json
# import requests
# import concurrent.futures
# from functools import wraps
# from flask import Flask, render_template, request, jsonify, session, redirect, url_for
# from pydub import AudioSegment
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)
# app.secret_key = os.getenv("SECRET_KEY", "fallback_secret_key")

# # --- Configuration ---
# ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin")
# ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "password")
# BASE_URL = os.getenv("API")

# # API Endpoints
# # BASE_URL = "https://includes-wales-candle-advertisers.trycloudflare.com" 
# LANG_API_URL = f"{BASE_URL}/M_predict_language"
# EMOTION_API_URL = f"{BASE_URL}/predict_emotion"
# TRANSCRIBE_API_URL = f"{BASE_URL}/transcribe"
# REFINE_API_URL = f"{BASE_URL}/refine-text"

# API_HEADERS = {
#   'Cookie': '.Tunnels.Relay.WebForwarding.Cookies=CfDJ8Cs4yarcs6pKkdu0hlKHsZuYEbzb1zuO1s_WPmIHLxLnGTkfcOo_twS4i1LL3t3PQKn0yu7bECL0OWzIE6ILVPfP3IoqjdZXMlolibBYdgiIUKnJglbrY6ZMnXxSN1sPyps0znVt-7TYrbVdtsCTh5RPKrTti95xbrT3nleH1AXda4UfrpyJ5pj6WyCfpRsn5xOJG1B1KUn5hUWkAdfiZyMbdbg8fyY_sKUhy3NpYlj4dJ28DXd9XX8ZgM05AFVWfcl86dpNHkA59Q4cgI5haJSFfEVS8tWFqkmIeo-q5Pm166XqwurtyUUb-Y15ZQqePP0BUlOr7W_TA2AieIjM1EoTAcXna6Xd0sIx37TClZkwR658kN42x5AVxn6ju52nWiDNJeUv-z4tzCISqsbUWgjNtVePuTPCUObOvZ_V0qm_ygDLqeH1RmB67wN-Eb7mEzTyf7JsUFrpFBgKFBUDKyb8MiAIzd5eXhH3dgcqPhPg6LH5dCXkVAYQLE7G-4l9dTIme6oa8x9pMalvSkg_OnBENpoRFKj8wf7ECb_j7xWtyFscTGEGQQlBQ85WIV4iOBog3zp6ql6bym6aTaAdHvp-le_puIbyVDiiqOFQXIHoBQ3dzkIFSY5kIiAZjDgaBWNiUMXBAdhGZZKb3U71QhUgb7dgIEroC7pQav6KA1aer8GxEqAOUjUMoSnDrGdY0k1ZMo8_jUKqf4GYzVEomyjN7iwsk4bblmbezZHpo_470JT0X3PuLktIsGxfBgEXXGFxV0lfUQeqfZaQkTHqPEl1SnvdG6tbpqU1_eg_J6zRXteu59PiNrxo_ff-uu3EQS-GAx8mRUg35WCw0sn0K1dPttxWFh0v-1QybOnZy-oy0yk16OfUK_7jFE_zZcFULMxUY30Nd3MCYuHq4y0w5YwMdFbxgIrB9hr2hST1ZMi6'
# }

# # --- Login Decorator ---
# def login_required(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         if 'logged_in' not in session:
#             return redirect(url_for('login'))
#         return f(*args, **kwargs)
#     return decorated_function

# # --- Routes: Auth ---
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         if username == ADMIN_USER and password == ADMIN_PASS:
#             session['logged_in'] = True
#             return redirect(url_for('index'))
#         else:
#             error = 'Invalid Credentials. Please try again.'
#     return render_template('login.html', error=error)

# @app.route('/logout')
# def logout():
#     session.pop('logged_in', None)
#     return redirect(url_for('login'))

# @app.route('/')
# @login_required
# def index():
#     return render_template('index.html')

# # --- Helper Functions ---
# def call_api_generic(url, audio_bytes, filename, content_type):
#     try:
#         files = [('audio_file', (filename, audio_bytes, content_type))]
#         response = requests.post(url, headers=API_HEADERS, files=files, timeout=30) 
#         if response.status_code == 200:
#             return response.json()
#         return {"error": f"HTTP {response.status_code}", "predicted_language": "Unknown", "predicted_emotion": "Unknown"}
#     except Exception as e:
#         print(f"API Fail {url}: {e}")
#         return {"error": str(e), "predicted_language": "Unknown", "predicted_emotion": "Unknown"}

# def call_transcribe_api(audio_bytes, filename):
#     try:
#         files = [('audio_file', (filename, audio_bytes, 'audio/wav'))]
#         response = requests.post(TRANSCRIBE_API_URL, files=files, timeout=60)
#         if response.status_code == 200:
#             return response.json()
#         return {"transcription": "", "error": f"HTTP {response.status_code}"}
#     except Exception as e:
#         print(f"Transcribe Fail: {e}")
#         return {"transcription": "", "error": str(e)}

# def call_refine_api(text):
#     if not text or len(text.strip()) == 0:
#         return {"detail": "Skipped (No text source)"}
        
#     try:
#         url = REFINE_API_URL
#         payload = json.dumps({"text": text})
#         headers = {'Content-Type': 'application/json'}
#         response = requests.post(url, headers=headers, data=payload, timeout=30)
        
#         if response.status_code == 200:
#             return response.json()
#         return {"detail": text, "error": f"Refine HTTP {response.status_code}"}
#     except Exception as e:
#         print(f"Refine Fail: {e}")
#         return {"detail": text, "error": str(e)}

# # --- Full Analysis Chunk Processor ---
# def process_chunk_full(chunk_audio, index, content_type):
#     """
#     Executes logic strictly in order: 
#     1. Emotion & Language (Audio Analysis)
#     2. Transcription & Refinement (Text Analysis)
#     """
#     buf = io.BytesIO()
#     chunk_audio.export(buf, format="wav")
#     chunk_bytes = buf.getvalue()
#     filename = f"chunk_{index}.wav"
    
#     start_time = index * 60
#     end_time = (index + 1) * 60
#     time_label = f"{start_time}m - {end_time}m"

#     # Containers
#     lang_res = {"predicted_language": "Unknown"}
#     emo_res = {"predicted_emotion": "Unknown"}
#     trans_res = {"transcription": ""}
#     refine_res = {"detail": ""}

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # === PHASE 1: Audio Analysis (Emotion & Lang) ===
#         future_lang = executor.submit(call_api_generic, LANG_API_URL, chunk_bytes, filename, "audio/wav")
#         future_emo = executor.submit(call_api_generic, EMOTION_API_URL, chunk_bytes, filename, "audio/wav")
        
#         try: lang_res = future_lang.result()
#         except Exception as e: lang_res = {"error": str(e), "predicted_language": "Unknown"}
            
#         try: emo_res = future_emo.result()
#         except Exception as e: emo_res = {"error": str(e), "predicted_emotion": "Unknown"}

#         # === PHASE 2: Text Analysis (Transcribe Only) ===
#         try:
#             trans_res = call_transcribe_api(chunk_bytes, filename)
#         except Exception as e:
#             trans_res = {"error": str(e), "transcription": ""}

#         # === PHASE 3: Refinement ===
#         raw_text = trans_res.get('transcription', '')
#         if raw_text:
#             try:
#                 refine_res = call_refine_api(raw_text)
#             except Exception as e:
#                 refine_res = {"detail": raw_text, "error": str(e)}
#         else:
#             refine_res = {"detail": "No transcription available to refine."}

#         return {
#             "index": index,
#             "timestamp": time_label,
#             "lang_res": lang_res,
#             "emo_res": emo_res,
#             "trans_res": trans_res,
#             "refine_res": refine_res
#         }

# # --- ROUTE: Full Analysis (Master) ---
# @app.route('/predict', methods=['POST'])
# @login_required
# def predict():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file provided'}), 400

#     audio_file = request.files['audio']
#     filename = audio_file.filename
#     content_type = audio_file.content_type 

#     try:
#         audio = AudioSegment.from_file(audio_file)
#         duration_sec = len(audio) / 1000.0

#         # === SINGLE FILE PROCESSING (< 60s) ===
#         if duration_sec <= 60:
#             audio_file.seek(0)
#             audio_bytes = audio_file.read()
            
#             # Default containers
#             lang_data = {"predicted_language": "Unknown"}
#             emo_data = {"predicted_emotion": "Unknown"}
#             trans_data = {"transcription": ""}
#             refine_data = {"detail": ""}

#             with concurrent.futures.ThreadPoolExecutor() as executor:
#                 # 1. Execute Audio Logic (Emotion & Lang)
#                 lang_fut = executor.submit(call_api_generic, LANG_API_URL, audio_bytes, filename, content_type)
#                 emo_fut = executor.submit(call_api_generic, EMOTION_API_URL, audio_bytes, filename, content_type)
                
#                 try: lang_data = lang_fut.result()
#                 except Exception: pass
                
#                 try: emo_data = emo_fut.result()
#                 except Exception: pass

#                 # 2. Execute Text Logic (Transcribe Only)
#                 try:
#                     trans_data = call_transcribe_api(audio_bytes, filename)
#                 except Exception: pass
                
#                 # 3. Refine Logic
#                 raw_text = trans_data.get('transcription', '')
#                 if raw_text:
#                     refine_data = call_refine_api(raw_text)
#                 else:
#                     refine_data = {"detail": "Refinement skipped (No text)"}

#                 return jsonify({
#                     "type": "single",
#                     "language_data": lang_data,
#                     "emotion_data": emo_data,
#                     "transcription_data": trans_data,
#                     "refinement_data": refine_data
#                 })

#         # === CHUNK PROCESSING (> 60s) ===
#         else:
#             chunk_length_ms = 60000 
#             chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            
#             results = []
#             with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#                 # Calls 'process_chunk_full' which strictly follows: Audio Logic -> Transcribe Logic
#                 futures = [executor.submit(process_chunk_full, chunk, i, content_type) for i, chunk in enumerate(chunks)]
#                 for future in concurrent.futures.as_completed(futures):
#                     try:
#                         results.append(future.result())
#                     except Exception as e:
#                         print(f"Chunk processing failed: {e}")
#                         results.append({
#                             "index": -1, "timestamp": "Error", 
#                             "lang_res": {"predicted_language": "Unknown"},
#                             "emo_res": {"predicted_emotion": "Unknown"}
#                         })
            
#             results.sort(key=lambda x: x['index'])

#             # --- Aggregation Logic ---
#             timeline_labels = [r['timestamp'] for r in results if r['index'] != -1]
#             emotion_timeline = [r['emo_res'].get('predicted_emotion', 'Unknown') for r in results]
            
#             lang_counts = {}
#             emotion_counts = {}
#             transcript_log = [] 

#             for r in results:
#                 if r['index'] == -1: continue 

#                 l = r['lang_res'].get('predicted_language', 'Unknown')
#                 lang_counts[l] = lang_counts.get(l, 0) + 1
                
#                 e = r['emo_res'].get('predicted_emotion', 'Unknown')
#                 emotion_counts[e] = emotion_counts.get(e, 0) + 1
                
#                 transcript_log.append({
#                     "timestamp": r['timestamp'],
#                     "text": r['trans_res'].get('transcription', 'Error/Empty'),
#                     "refined": r['refine_res'].get('detail', 'Error/Empty')
#                 })

#             return jsonify({
#                 "type": "report",
#                 "duration": duration_sec,
#                 "timeline": {
#                     "labels": timeline_labels,
#                     "emotions": emotion_timeline
#                 },
#                 "stats": {
#                     "languages": lang_counts,
#                     "emotions": emotion_counts
#                 },
#                 "transcripts": transcript_log
#             })

#     except Exception as e:
#         print(f"Global Error: {e}")
#         return jsonify({"error": "Failed to process audio file."}), 500

# # --- ROUTE: Transcribe Only ---
# @app.route('/transcribe_only', methods=['POST'])
# @login_required
# def transcribe_only():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file provided'}), 400

#     audio_file = request.files['audio']
#     filename = audio_file.filename
    
#     try:
#         audio_file.seek(0)
#         audio_bytes = audio_file.read()

#         # Step 1: Transcribe
#         trans_data = call_transcribe_api(audio_bytes, filename)
        
#         # Step 2: Refine
#         raw_text = trans_data.get('transcription', '')
#         refine_data = {"detail": "Refinement skipped (No text)"}
        
#         if raw_text:
#             refine_data = call_refine_api(raw_text)

#         return jsonify({
#             "type": "single",
#             "language_data": {"predicted_language": "--"},
#             "emotion_data": {"predicted_emotion": "--", "confidence_percent": 0},
#             "transcription_data": trans_data,
#             "refinement_data": refine_data
#         })

#     except Exception as e:
#         print(f"Transcribe Only Error: {e}")
#         return jsonify({"error": "Failed to transcribe file."}), 500

# # --- ROUTE: Audio Only (Emotion & Lang) ---
# @app.route('/audio_only', methods=['POST'])
# @login_required
# def audio_only():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file provided'}), 400

#     audio_file = request.files['audio']
#     filename = audio_file.filename
#     content_type = audio_file.content_type
    
#     try:
#         audio = AudioSegment.from_file(audio_file)
#         duration_sec = len(audio) / 1000.0

#         # === 1. SINGLE FILE LOGIC (< 60s) ===
#         if duration_sec <= 60:
#             audio_file.seek(0)
#             audio_bytes = audio_file.read()
            
#             lang_data = {"predicted_language": "Unknown"}
#             emo_data = {"predicted_emotion": "Unknown"}

#             # Run Language & Emotion in Parallel
#             with concurrent.futures.ThreadPoolExecutor() as executor:
#                 lang_fut = executor.submit(call_api_generic, LANG_API_URL, audio_bytes, filename, content_type)
#                 emo_fut = executor.submit(call_api_generic, EMOTION_API_URL, audio_bytes, filename, content_type)
                
#                 try: lang_data = lang_fut.result()
#                 except Exception: pass
                
#                 try: emo_data = emo_fut.result()
#                 except Exception: pass

#             return jsonify({
#                 "type": "single",
#                 "language_data": lang_data,
#                 "emotion_data": emo_data,
#                 "transcription_data": {"transcription": "Skipped (Audio Analysis Only)"},
#                 "refinement_data": {"detail": "Skipped (Audio Analysis Only)"}
#             })

#         # === 2. CHUNK LOGIC (> 60s) ===
#         else:
#             chunk_length_ms = 60000 
#             chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            
#             results = []
            
#             # Helper for Audio Only processing
#             def process_chunk_audio_only(chunk_segment, idx):
#                 buf = io.BytesIO()
#                 chunk_segment.export(buf, format="wav")
#                 c_bytes = buf.getvalue()
#                 c_name = f"chunk_{idx}.wav"
#                 time_lbl = f"{idx * 60}m - {(idx + 1) * 60}m"
                
#                 l_res = {"predicted_language": "Unknown"}
#                 e_res = {"predicted_emotion": "Unknown"}
                
#                 with concurrent.futures.ThreadPoolExecutor() as exc:
#                     f_l = exc.submit(call_api_generic, LANG_API_URL, c_bytes, c_name, "audio/wav")
#                     f_e = exc.submit(call_api_generic, EMOTION_API_URL, c_bytes, c_name, "audio/wav")
#                     try: l_res = f_l.result()
#                     except: pass
#                     try: e_res = f_e.result()
#                     except: pass
                
#                 return {
#                     "index": idx,
#                     "timestamp": time_lbl,
#                     "lang_res": l_res,
#                     "emo_res": e_res
#                 }

#             with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#                 futures = [executor.submit(process_chunk_audio_only, chunk, i) for i, chunk in enumerate(chunks)]
#                 for future in concurrent.futures.as_completed(futures):
#                     try:
#                         results.append(future.result())
#                     except Exception as e:
#                         print(f"Chunk audio-only failed: {e}")
#                         results.append({
#                             "index": -1, "timestamp": "Error", 
#                             "lang_res": {"predicted_language": "Unknown"}, 
#                             "emo_res": {"predicted_emotion": "Unknown"}
#                         })

#             results.sort(key=lambda x: x['index'])

#             timeline_labels = [r['timestamp'] for r in results if r['index'] != -1]
#             emotion_timeline = [r['emo_res'].get('predicted_emotion', 'Unknown') for r in results]
            
#             lang_counts = {}
#             emotion_counts = {}
#             transcript_log = [] 

#             for r in results:
#                 if r['index'] == -1: continue 
#                 l = r['lang_res'].get('predicted_language', 'Unknown')
#                 lang_counts[l] = lang_counts.get(l, 0) + 1
#                 e = r['emo_res'].get('predicted_emotion', 'Unknown')
#                 emotion_counts[e] = emotion_counts.get(e, 0) + 1

#             return jsonify({
#                 "type": "report",
#                 "duration": duration_sec,
#                 "timeline": {
#                     "labels": timeline_labels,
#                     "emotions": emotion_timeline
#                 },
#                 "stats": {
#                     "languages": lang_counts,
#                     "emotions": emotion_counts
#                 },
#                 "transcripts": transcript_log 
#             })

#     except Exception as e:
#         print(f"Audio Only Error: {e}")
#         return jsonify({"error": "Failed to process audio."}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)



import os
import io
import json
import requests
import concurrent.futures
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from pydub import AudioSegment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback_secret_key")

# --- Configuration ---
ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "password")
BASE_URL = os.getenv("API")

# API Endpoints
# BASE_URL = "https://includes-wales-candle-advertisers.trycloudflare.com" 
LANG_API_URL = f"{BASE_URL}/M_predict_language"
EMOTION_API_URL = f"{BASE_URL}/predict_emotion"
TRANSCRIBE_API_URL = f"{BASE_URL}/transcribe"
REFINE_API_URL = f"{BASE_URL}/refine-text"

API_HEADERS = {
  'Cookie': '.Tunnels.Relay.WebForwarding.Cookies=CfDJ8Cs4yarcs6pKkdu0hlKHsZuYEbzb1zuO1s_WPmIHLxLnGTkfcOo_twS4i1LL3t3PQKn0yu7bECL0OWzIE6ILVPfP3IoqjdZXMlolibBYdgiIUKnJglbrY6ZMnXxSN1sPyps0znVt-7TYrbVdtsCTh5RPKrTti95xbrT3nleH1AXda4UfrpyJ5pj6WyCfpRsn5xOJG1B1KUn5hUWkAdfiZyMbdbg8fyY_sKUhy3NpYlj4dJ28DXd9XX8ZgM05AFVWfcl86dpNHkA59Q4cgI5haJSFfEVS8tWFqkmIeo-q5Pm166XqwurtyUUb-Y15ZQqePP0BUlOr7W_TA2AieIjM1EoTAcXna6Xd0sIx37TClZkwR658kN42x5AVxn6ju52nWiDNJeUv-z4tzCISqsbUWgjNtVePuTPCUObOvZ_V0qm_ygDLqeH1RmB67wN-Eb7mEzTyf7JsUFrpFBgKFBUDKyb8MiAIzd5eXhH3dgcqPhPg6LH5dCXkVAYQLE7G-4l9dTIme6oa8x9pMalvSkg_OnBENpoRFKj8wf7ECb_j7xWtyFscTGEGQQlBQ85WIV4iOBog3zp6ql6bym6aTaAdHvp-le_puIbyVDiiqOFQXIHoBQ3dzkIFSY5kIiAZjDgaBWNiUMXBAdhGZZKb3U71QhUgb7dgIEroC7pQav6KA1aer8GxEqAOUjUMoSnDrGdY0k1ZMo8_jUKqf4GYzVEomyjN7iwsk4bblmbezZHpo_470JT0X3PuLktIsGxfBgEXXGFxV0lfUQeqfZaQkTHqPEl1SnvdG6tbpqU1_eg_J6zRXteu59PiNrxo_ff-uu3EQS-GAx8mRUg35WCw0sn0K1dPttxWFh0v-1QybOnZy-oy0yk16OfUK_7jFE_zZcFULMxUY30Nd3MCYuHq4y0w5YwMdFbxgIrB9hr2hST1ZMi6'
}

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
        if username == ADMIN_USER and password == ADMIN_PASS:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = 'Invalid Credentials. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

# --- Helper Functions ---
def call_api_generic(url, audio_bytes, filename, content_type):
    try:
        files = [('audio_file', (filename, audio_bytes, content_type))]
        response = requests.post(url, headers=API_HEADERS, files=files, timeout=30) 
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}", "predicted_language": "Unknown", "predicted_emotion": "Unknown"}
    except Exception as e:
        print(f"API Fail {url}: {e}")
        return {"error": str(e), "predicted_language": "Unknown", "predicted_emotion": "Unknown"}

def call_transcribe_api(audio_bytes, filename):
    try:
        files = [('audio_file', (filename, audio_bytes, 'audio/wav'))]
        response = requests.post(TRANSCRIBE_API_URL, files=files, timeout=120) # Increased timeout for large files
        if response.status_code == 200:
            return response.json()
        return {"transcription": "", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        print(f"Transcribe Fail: {e}")
        return {"transcription": "", "error": str(e)}

def call_refine_api(text):
    if not text or len(text.strip()) == 0:
        return {"detail": "Skipped (No text source)"}
        
    try:
        url = REFINE_API_URL
        payload = json.dumps({"text": text})
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        return {"detail": text, "error": f"Refine HTTP {response.status_code}"}
    except Exception as e:
        print(f"Refine Fail: {e}")
        return {"detail": text, "error": str(e)}

# --- Full Analysis Chunk Processor (Only Emotion & Language) ---
def process_chunk_audio_only(chunk_audio, index, content_type):
    """
    Processes chunks ONLY for Emotion and Language distribution.
    Transcription is handled globally.
    """
    buf = io.BytesIO()
    chunk_audio.export(buf, format="wav")
    chunk_bytes = buf.getvalue()
    filename = f"chunk_{index}.wav"
    
    start_time = index * 60
    end_time = (index + 1) * 60
    time_label = f"{start_time}m - {end_time}m"

    lang_res = {"predicted_language": "Unknown"}
    emo_res = {"predicted_emotion": "Unknown"}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_lang = executor.submit(call_api_generic, LANG_API_URL, chunk_bytes, filename, "audio/wav")
        future_emo = executor.submit(call_api_generic, EMOTION_API_URL, chunk_bytes, filename, "audio/wav")
        
        try: lang_res = future_lang.result()
        except Exception as e: lang_res = {"error": str(e), "predicted_language": "Unknown"}
            
        try: emo_res = future_emo.result()
        except Exception as e: emo_res = {"error": str(e), "predicted_emotion": "Unknown"}

    return {
        "index": index,
        "timestamp": time_label,
        "lang_res": lang_res,
        "emo_res": emo_res
    }

# --- ROUTE: Full Analysis (Master) ---
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    filename = audio_file.filename
    content_type = audio_file.content_type 

    try:
        # 1. Read entire file bytes ONCE for global transcription
        audio_file.seek(0)
        full_audio_bytes = audio_file.read()

        audio = AudioSegment.from_file(io.BytesIO(full_audio_bytes))
        duration_sec = len(audio) / 1000.0

        # === A. PREPARE TRANSCRIPTION (GLOBAL) ===
        # We start this immediately. It can run in parallel with chunk processing.
        
        def run_full_transcription():
            t_data = call_transcribe_api(full_audio_bytes, filename)
            raw_text = t_data.get('transcription', '')
            r_data = {"detail": "Refinement skipped (No text)"}
            if raw_text:
                r_data = call_refine_api(raw_text)
            return t_data, r_data

        # === B. PROCESS CHUNKS (EMOTION & LANG) ===
        # Regardless of length, if it's > 60s we chunk for stats. If < 60s, we treat as 1 chunk.
        
        chunk_length_ms = 60000 
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        # Use ThreadPool to run Chunk Analysis AND Full Transcription in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 1. Submit Transcription Task (Entire File)
            transcribe_future = executor.submit(run_full_transcription)

            # 2. Submit Chunk Analysis Tasks (Emotion/Lang per minute)
            chunk_futures = [executor.submit(process_chunk_audio_only, chunk, i, content_type) for i, chunk in enumerate(chunks)]
            
            # 3. Gather Results
            chunk_results = []
            for future in concurrent.futures.as_completed(chunk_futures):
                try:
                    chunk_results.append(future.result())
                except Exception as e:
                    print(f"Chunk failed: {e}")
                    chunk_results.append({
                        "index": -1, "timestamp": "Error", 
                        "lang_res": {"predicted_language": "Unknown"}, 
                        "emo_res": {"predicted_emotion": "Unknown"}
                    })
            
            # Wait for transcription
            trans_data, refine_data = transcribe_future.result()

        # === C. AGGREGATE RESULTS ===
        chunk_results.sort(key=lambda x: x['index'])

        # 1. Single File View (If only 1 chunk / < 60s)
        if duration_sec <= 60:
            # Extract first chunk data for the single view cards
            first_chunk = chunk_results[0] if chunk_results else {}
            return jsonify({
                "type": "single",
                "language_data": first_chunk.get('lang_res', {}),
                "emotion_data": first_chunk.get('emo_res', {}),
                "transcription_data": trans_data,
                "refinement_data": refine_data
            })

        # 2. Report View (> 60s)
        else:
            timeline_labels = [r['timestamp'] for r in chunk_results if r['index'] != -1]
            emotion_timeline = [r['emo_res'].get('predicted_emotion', 'Unknown') for r in chunk_results]
            
            lang_counts = {}
            emotion_counts = {}

            for r in chunk_results:
                if r['index'] == -1: continue 
                l = r['lang_res'].get('predicted_language', 'Unknown')
                lang_counts[l] = lang_counts.get(l, 0) + 1
                e = r['emo_res'].get('predicted_emotion', 'Unknown')
                emotion_counts[e] = emotion_counts.get(e, 0) + 1

            # Create a SINGLE entry for the transcript log containing the full text
            transcript_log = [{
                "timestamp": "Full Document",
                "text": trans_data.get('transcription', 'Error/Empty'),
                "refined": refine_data.get('detail', 'Error/Empty')
            }]

            return jsonify({
                "type": "report",
                "duration": duration_sec,
                "timeline": {
                    "labels": timeline_labels,
                    "emotions": emotion_timeline
                },
                "stats": {
                    "languages": lang_counts,
                    "emotions": emotion_counts
                },
                "transcripts": transcript_log
            })

    except Exception as e:
        print(f"Global Error: {e}")
        return jsonify({"error": "Failed to process audio file."}), 500

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

        # Step 1: Transcribe Full File
        trans_data = call_transcribe_api(audio_bytes, filename)
        
        # Step 2: Refine Full File
        raw_text = trans_data.get('transcription', '')
        refine_data = {"detail": "Refinement skipped (No text)"}
        
        if raw_text:
            refine_data = call_refine_api(raw_text)

        return jsonify({
            "type": "single",
            "language_data": {"predicted_language": "--"},
            "emotion_data": {"predicted_emotion": "--", "confidence_percent": 0},
            "transcription_data": trans_data,
            "refinement_data": refine_data
        })

    except Exception as e:
        print(f"Transcribe Only Error: {e}")
        return jsonify({"error": "Failed to transcribe file."}), 500

# --- ROUTE: Audio Only (Emotion & Lang) ---
@app.route('/audio_only', methods=['POST'])
@login_required
def audio_only():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    filename = audio_file.filename
    content_type = audio_file.content_type
    
    try:
        audio = AudioSegment.from_file(audio_file)
        duration_sec = len(audio) / 1000.0

        # Logic for Audio Only remains chunk-based for stats, or single for short
        # Just no transcription call.
        
        if duration_sec <= 60:
            audio_file.seek(0)
            audio_bytes = audio_file.read()
            lang_data = {"predicted_language": "Unknown"}
            emo_data = {"predicted_emotion": "Unknown"}

            with concurrent.futures.ThreadPoolExecutor() as executor:
                lang_fut = executor.submit(call_api_generic, LANG_API_URL, audio_bytes, filename, content_type)
                emo_fut = executor.submit(call_api_generic, EMOTION_API_URL, audio_bytes, filename, content_type)
                try: lang_data = lang_fut.result()
                except: pass
                try: emo_data = emo_fut.result()
                except: pass

            return jsonify({
                "type": "single",
                "language_data": lang_data,
                "emotion_data": emo_data,
                "transcription_data": {"transcription": "Skipped (Audio Analysis Only)"},
                "refinement_data": {"detail": "Skipped (Audio Analysis Only)"}
            })

        else:
            chunk_length_ms = 60000 
            chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_chunk_audio_only, chunk, i, content_type) for i, chunk in enumerate(chunks)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception:
                        results.append({"index": -1}) # Simplified error handling for brevity

            results.sort(key=lambda x: x['index'])
            
            timeline_labels = [r['timestamp'] for r in results if r['index'] != -1]
            emotion_timeline = [r['emo_res'].get('predicted_emotion', 'Unknown') for r in results if r['index'] != -1]
            
            lang_counts = {}
            emotion_counts = {}

            for r in results:
                if r['index'] == -1: continue 
                l = r['lang_res'].get('predicted_language', 'Unknown')
                lang_counts[l] = lang_counts.get(l, 0) + 1
                e = r['emo_res'].get('predicted_emotion', 'Unknown')
                emotion_counts[e] = emotion_counts.get(e, 0) + 1

            return jsonify({
                "type": "report",
                "duration": duration_sec,
                "timeline": {"labels": timeline_labels, "emotions": emotion_timeline},
                "stats": {"languages": lang_counts, "emotions": emotion_counts},
                "transcripts": [] # Empty for Audio Only
            })

    except Exception as e:
        print(f"Audio Only Error: {e}")
        return jsonify({"error": "Failed to process audio."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)