import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
WEIGHTS_PATH = r'D:\FUSION_TEST\best_heart_model_final.keras'
CSV_PATH = r'D:\DATASET DOWNLOADS\EchoNet-Dynamic 7.4GB avi files\FileList.csv'
VIDEO_DIR = r'D:\DATASET DOWNLOADS\EchoNet-Dynamic 7.4GB avi files\Videos'
IMG_SIZE = 112
NUM_FRAMES = 16
NUM_SAMPLES = 20 # Using 20 samples for a reliable audit check

def build_heart_model():
    img_shape = (IMG_SIZE, IMG_SIZE, 3)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_shape, 
        include_top=False, 
        weights=None # We will load our own weights
    )
    
    inputs = layers.Input(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    x = layers.TimeDistributed(base_model)(inputs)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.LSTM(64, return_sequences=False, name="temporal_memory")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='linear', name="ef_prediction")(x)
    
    model = models.Model(inputs, outputs, name="HepaCardio_Heart_v1")
    return model

def _load_video_sequence(path, img_size=112, num_frames=16):
    cap = cv2.VideoCapture(path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return None
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if idx in indices:
            frame = cv2.resize(frame, (img_size, img_size))
            frame = frame / 255.0 
            frames.append(frame)
        if len(frames) == num_frames: break
    cap.release()
    
    if frames and len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])
    elif not frames:
        return None
        
    return np.array(frames)

def evaluate():
    print("Reconstructing heart model architecture...")
    model = build_heart_model()
    
    print(f"Loading weights from {WEIGHTS_PATH}...")
    # Using skip_mismatch=True in case of slight version differences in layer naming
    try:
        model.load_weights(WEIGHTS_PATH)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Standard weight load failed: {e}")
        print("Attempting to load as Keras 3 model directly...")
        try:
            # Fallback to direct load if reconstruction fails due to naming
            model = tf.keras.models.load_model(WEIGHTS_PATH, compile=False)
            print("Direct load successful via Keras 3 loader.")
        except Exception as e2:
            print(f"All load methods failed. Error: {e2}")
            return

    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    test_df = df[df['Split'] == 'TEST'].sample(n=NUM_SAMPLES, random_state=42)
    
    predictions = []
    actuals = []
    
    for idx, row in test_df.iterrows():
        video_path = os.path.join(VIDEO_DIR, str(row['FileName']) + ".avi")
        if not os.path.exists(video_path):
            continue
            
        frames = _load_video_sequence(video_path, IMG_SIZE, NUM_FRAMES)
        if frames is not None:
            frames = np.expand_dims(frames, axis=0)
            pred = model.predict(frames, verbose=0)[0][0]
            predictions.append(pred)
            actuals.append(row['EF'])
            
            mae_current = mean_absolute_error(actuals, predictions)
            print(f"[{len(predictions)}/{NUM_SAMPLES}] Filename: {row['FileName']} | Pred: {pred:.2f}% | Actual: {row['EF']}% | Running MAE: {mae_current:.4f}")

    final_mae = mean_absolute_error(actuals, predictions)
    print("\n" + "="*30)
    print(f"HEART MODEL AUDIT COMPLETE")
    print(f"FINAL MAE: {final_mae:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()
