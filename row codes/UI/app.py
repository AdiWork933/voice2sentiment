import os
import io
import requests
import concurrent.futures
from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment

app = Flask(__name__)

# --- Configuration ---
BASE_URL = "http://127.0.0.1:8000"
LANG_API_URL = f"{BASE_URL}/M_predict_language"
EMOTION_API_URL = f"{BASE_URL}/predict_emotion"

API_HEADERS = {
  'Cookie': '.Tunnels.Relay.WebForwarding.Cookies=CfDJ8Cs4yarcs6pKkdu0hlKHsZuYEbzb1zuO1s_WPmIHLxLnGTkfcOo_twS4i1LL3t3PQKn0yu7bECL0OWzIE6ILVPfP3IoqjdZXMlolibBYdgiIUKnJglbrY6ZMnXxSN1sPyps0znVt-7TYrbVdtsCTh5RPKrTti95xbrT3nleH1AXda4UfrpyJ5pj6WyCfpRsn5xOJG1B1KUn5hUWkAdfiZyMbdbg8fyY_sKUhy3NpYlj4dJ28DXd9XX8ZgM05AFVWfcl86dpNHkA59Q4cgI5haJSFfEVS8tWFqkmIeo-q5Pm166XqwurtyUUb-Y15ZQqePP0BUlOr7W_TA2AieIjM1EoTAcXna6Xd0sIx37TClZkwR658kN42x5AVxn6ju52nWiDNJeUv-z4tzCISqsbUWgjNtVePuTPCUObOvZ_V0qm_ygDLqeH1RmB67wN-Eb7mEzTyf7JsUFrpFBgKFBUDKyb8MiAIzd5eXhH3dgcqPhPg6LH5dCXkVAYQLE7G-4l9dTIme6oa8x9pMalvSkg_OnBENpoRFKj8wf7ECb_j7xWtyFscTGEGQQlBQ85WIV4iOBog3zp6ql6bym6aTaAdHvp-le_puIbyVDiiqOFQXIHoBQ3dzkIFSY5kIiAZjDgaBWNiUMXBAdhGZZKb3U71QhUgb7dgIEroC7pQav6KA1aer8GxEqAOUjUMoSnDrGdY0k1ZMo8_jUKqf4GYzVEomyjN7iwsk4bblmbezZHpo_470JT0X3PuLktIsGxfBgEXXGFxV0lfUQeqfZaQkTHqPEl1SnvdG6tbpqU1_eg_J6zRXteu59PiNrxo_ff-uu3EQS-GAx8mRUg35WCw0sn0K1dPttxWFh0v-1QybOnZy-oy0yk16OfUK_7jFE_zZcFULMxUY30Nd3MCYuHq4y0w5YwMdFbxgIrB9hr2hST1ZMi6'
}

@app.route('/')
def index():
    return render_template('index.html')

def call_api(url, audio_bytes, filename, content_type):
    try:
        files = [('audio_file', (filename, audio_bytes, content_type))]
        response = requests.post(url, headers=API_HEADERS, files=files, timeout=120)
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error {response.status_code}", "raw": response.text}
    except Exception as e:
        return {"error": str(e)}

def process_chunk(chunk_audio, index, content_type):
    buf = io.BytesIO()
    chunk_audio.export(buf, format="wav")
    chunk_bytes = buf.getvalue()
    filename = f"chunk_{index}.wav"
    
    start_time = index * 60
    end_time = (index + 1) * 60
    time_label = f"{start_time}m - {end_time}m" # Labeling as minutes roughly

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_lang = executor.submit(call_api, LANG_API_URL, chunk_bytes, filename, "audio/wav")
        future_emo = executor.submit(call_api, EMOTION_API_URL, chunk_bytes, filename, "audio/wav")
        
        return {
            "index": index,
            "timestamp": time_label,
            "lang_res": future_lang.result(),
            "emo_res": future_emo.result()
        }

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    filename = audio_file.filename
    content_type = audio_file.content_type 

    try:
        audio = AudioSegment.from_file(audio_file)
        duration_sec = len(audio) / 1000.0

        if duration_sec <= 60:
            audio_file.seek(0)
            audio_bytes = audio_file.read()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                lang_fut = executor.submit(call_api, LANG_API_URL, audio_bytes, filename, content_type)
                emo_fut = executor.submit(call_api, EMOTION_API_URL, audio_bytes, filename, content_type)
                return jsonify({
                    "type": "single",
                    "language_data": lang_fut.result(),
                    "emotion_data": emo_fut.result()
                })
        else:
            chunk_length_ms = 60000 
            chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_chunk, chunk, i, content_type) for i, chunk in enumerate(chunks)]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            
            results.sort(key=lambda x: x['index'])

            # --- Aggregation Logic ---
            timeline_labels = [r['timestamp'] for r in results]
            emotion_timeline = [r['emo_res'].get('predicted_emotion', 'Unknown') for r in results]
            
            lang_counts = {}
            emotion_counts = {} # New Dictionary for Emotion Stats

            for r in results:
                # Count Languages
                l = r['lang_res'].get('predicted_language', 'Unknown')
                lang_counts[l] = lang_counts.get(l, 0) + 1
                
                # Count Emotions
                e = r['emo_res'].get('predicted_emotion', 'Unknown')
                emotion_counts[e] = emotion_counts.get(e, 0) + 1

            return jsonify({
                "type": "report",
                "duration": duration_sec,
                "timeline": {
                    "labels": timeline_labels,
                    "emotions": emotion_timeline
                },
                "stats": {
                    "languages": lang_counts,
                    "emotions": emotion_counts  # Sent to frontend
                }
            })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to process audio."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)