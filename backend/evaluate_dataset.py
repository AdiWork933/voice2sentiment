import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

# Import your existing models
from emotion_model import EmotionPredictor
from language_model import MultiLanguagePredictor

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_ROOT = "E://Audio_D"

def main():
    # Start the master stopwatch
    pipeline_start_time = time.time()
    
    # 1. Load Environment Variables (for model paths)
    load_dotenv()
    
    # 2. Initialize Models
    print("\n" + "="*50)
    print("🚀 INITIALIZING MODELS FOR EVALUATION")
    print("="*50)
    
    emotion_predictor = EmotionPredictor(os.getenv("EMOTION_MODEL_PATH"))
    
    multi_model_paths = {
        'hindi': os.getenv("HINDI_MODEL_PATH"),
        'english': os.getenv("ENGLISH_MODEL_PATH"),
        'bengali': os.getenv("BENGALI_MODEL_PATH")
    }
    multi_lang_predictor = MultiLanguagePredictor(multi_model_paths)
    
    # 3. Setup Data Collection
    results = []
    
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Error: Dataset path not found at {DATASET_ROOT}")
        return

    # 4. Crawl the Dataset
    languages = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    
    print("\n" + "="*50)
    print("🎧 STARTING DATASET EVALUATION")
    print("="*50)
    
    for lang in languages:
        lang_path = os.path.join(DATASET_ROOT, lang)
        emotions = [d for d in os.listdir(lang_path) if os.path.isdir(os.path.join(lang_path, d))]
        
        for emotion in emotions:
            emo_path = os.path.join(lang_path, emotion)
            audio_files = [f for f in os.listdir(emo_path) if f.endswith(('.wav', '.mp3'))]
            
            for audio_file in tqdm(audio_files, desc=f"Processing {lang.capitalize()}/{emotion.capitalize()}"):
                file_path = os.path.join(emo_path, audio_file)
                
                # --- Get Predictions & Times ---
                pred_emo, emo_conf, emo_time = emotion_predictor.predict(file_path)
                pred_lang, lang_time = multi_lang_predictor.predict(file_path)
                
                # --- Store Results ---
                results.append({
                    'file': audio_file,
                    'true_language': lang.strip().lower(),
                    'pred_language': pred_lang.strip().lower() if pred_lang else "undetermined",
                    'true_emotion': emotion.strip().lower(),
                    'pred_emotion': pred_emo.strip().lower() if pred_emo else "unknown",
                    'emo_time': emo_time,
                    'lang_time': lang_time,
                    'total_sample_time': emo_time + lang_time
                })

    # Stop the master stopwatch
    pipeline_end_time = time.time()
    total_execution_time = pipeline_end_time - pipeline_start_time

    # 5. Generate Reports
    if not results:
        print("⚠️ No audio files found to evaluate.")
        return

    df = pd.DataFrame(results)
    
    print("\n\n" + "="*50)
    print("📊 EVALUATION REPORT")
    print("="*50)
    
    # --- OVERALL TIME SUMMARY ---
    print("\n⏱️ TIME TRACKING SUMMARY")
    print("-" * 40)
    print(f"Total Files Processed: {len(df)}")
    print(f"Total Execution Time (including loading): {total_execution_time:.2f} seconds")
    print(f"Total Time spent purely predicting (Emotion): {df['emo_time'].sum():.2f} seconds")
    print(f"Total Time spent purely predicting (Language): {df['lang_time'].sum():.2f} seconds")
    print(f"Average Time per File (Combined Models): {df['total_sample_time'].mean():.4f} seconds")

    # --- LANGUAGE REPORT ---
    print("\n🗣️ LANGUAGE DETECTION PERFORMANCE")
    print("-" * 40)
    print(classification_report(df['true_language'], df['pred_language'], zero_division=0))
    print(f"Overall Language Accuracy: {accuracy_score(df['true_language'], df['pred_language']) * 100:.2f}%")

    # --- EMOTION REPORT ---
    print("\n🎭 EMOTION DETECTION PERFORMANCE")
    print("-" * 40)
    print(classification_report(df['true_emotion'], df['pred_emotion'], zero_division=0))
    print(f"Overall Emotion Accuracy: {accuracy_score(df['true_emotion'], df['pred_emotion']) * 100:.2f}%")

    # 6. Save detailed results to CSV
    csv_path = "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    
    # 7. Generate a 4-Panel Time Tracking Dashboard
    print("\n📈 Generating Time Analysis Dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Prediction Time Analysis', fontsize=18, fontweight='bold', y=0.98)

    # Top-Left: Sequential Tracking
    axes[0, 0].plot(df.index, df['emo_time'], label='Emotion Time', color='blue', alpha=0.7)
    axes[0, 0].plot(df.index, df['lang_time'], label='Language Time', color='orange', alpha=0.7)
    axes[0, 0].set_title('1. Time per Single Data (Sequential)', fontsize=14)
    axes[0, 0].set_xlabel('Sample Index (File Number)')
    axes[0, 0].set_ylabel('Prediction Time (seconds)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # Top-Right: Time Distribution (Histogram)
    axes[0, 1].hist(df['emo_time'], bins=30, color='blue', alpha=0.5, label='Emotion')
    axes[0, 1].hist(df['lang_time'], bins=30, color='orange', alpha=0.5, label='Language')
    axes[0, 1].set_title('2. Distribution of Prediction Times', fontsize=14)
    axes[0, 1].set_xlabel('Prediction Time (seconds)')
    axes[0, 1].set_ylabel('Number of Files')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # Bottom-Left: Average Time by Emotion Class
    emo_avg = df.groupby('true_emotion')['emo_time'].mean()
    emo_avg.plot(kind='bar', ax=axes[1, 0], color='cornflowerblue', edgecolor='black')
    axes[1, 0].set_title('3. Avg Emotion Prediction Time by Class', fontsize=14)
    axes[1, 0].set_xlabel('Emotion Class')
    axes[1, 0].set_ylabel('Avg Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)

    # Bottom-Right: Average Time by Language Class
    lang_avg = df.groupby('true_language')['lang_time'].mean()
    lang_avg.plot(kind='bar', ax=axes[1, 1], color='sandybrown', edgecolor='black')
    axes[1, 1].set_title('4. Avg Language Prediction Time by Class', fontsize=14)
    axes[1, 1].set_xlabel('Language Class')
    axes[1, 1].set_ylabel('Avg Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.5)

    # Save the Dashboard
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to not cut off title
    graph_path = "time_analysis_dashboard.png"
    plt.savefig(graph_path, dpi=300)
    plt.close()
    
    print(f"📊 Dashboard successfully saved to {graph_path}")
    print("="*50)

if __name__ == "__main__":
    main()