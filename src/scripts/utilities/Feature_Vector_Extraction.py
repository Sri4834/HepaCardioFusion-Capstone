import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model, Model

# --- CONFIGURATION ---
VIDEO_DIR = "Videos/"
LIVER_DIR = "D:/New folder/UniqueLiverDataset/"
CSV_PATH = "fusion_training_dataset.csv"
HEART_MODEL_PATH = "best_heart_model_final.keras"
LIVER_MODEL_PATH = "liver_models/liver_model_optimized.keras"
HEART_FEATURE_LAYER = "temporal_memory"
LIVER_FEATURE_LAYER = "dense"
BATCH_SIZE = 32

# Create Output Folders
os.makedirs("heart_vectors", exist_ok=True)
os.makedirs("liver_vectors", exist_ok=True)

# --- 1. LOAD MODELS & STRIP OUTPUT HEADS ---
print("🚀 Loading Models...")
heart_full = load_model(HEART_MODEL_PATH)
keras.config.enable_unsafe_deserialization()
liver_full = load_model(LIVER_MODEL_PATH, safe_mode=False)

def build_feature_extractor(model, layer_name, model_label):
    try:
        layer_output = model.get_layer(layer_name).output
    except ValueError as err:
        layer_names = [layer.name for layer in model.layers]
        raise ValueError(
            f"{model_label}: feature layer '{layer_name}' not found. "
            f"Available layers: {layer_names}"
        ) from err
    return Model(inputs=model.input, outputs=layer_output)

heart_extractor = build_feature_extractor(heart_full, HEART_FEATURE_LAYER, "Heart model")
liver_extractor = build_feature_extractor(liver_full, LIVER_FEATURE_LAYER, "Liver model")

# --- 2. HEART VIDEO EXTRACTION ---
def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 16: return None
    
    indices = np.linspace(0, total_frames - 1, 16, dtype=int)
    for i in range(total_frames):
        ret, frame = cap.read()
        if i in indices and ret:
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame / 255.0)
    cap.release()
    
    if len(frames) < 16: return None
    # Add batch dimension: (1, 16, 224, 224, 3)
    input_data = np.expand_dims(np.array(frames), axis=0)
    return heart_extractor.predict(input_data, verbose=0)

print("🎥 Extracting Heart Vectors (this takes time) Please wait raja...")
df = pd.read_csv(CSV_PATH)

def resolve_video_path(video_filename):
    candidates = [
        os.path.join(VIDEO_DIR, str(video_filename)),
        os.path.join(VIDEO_DIR, f"{video_filename}.avi"),
        os.path.join(VIDEO_DIR, f"{video_filename}.mp4"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

heart_rows = df[["video_filename", "heart_feature_key"]].dropna().drop_duplicates()
for _, row in heart_rows.iterrows():
    video_filename = str(row["video_filename"])
    feature_key = str(row["heart_feature_key"])
    path = resolve_video_path(video_filename)
    if path:
        vector = extract_video_features(path)
        if vector is not None:
            np.save(os.path.join("heart_vectors", f"{feature_key}.npy"), vector.flatten())

# --- 3. LIVER IMAGE EXTRACTION ---
print("📸 Extracting Liver Vectors...")
img_map = {}
for dir_path, _, file_names in os.walk(LIVER_DIR):
    for file_name in file_names:
        root_name, _ = os.path.splitext(file_name)
        full_path = os.path.join(dir_path, file_name)
        img_map[root_name] = full_path

liver_rows = df[["liver_feature_key"]].dropna().drop_duplicates()
valid_items = []
for _, row in liver_rows.iterrows():
    key = str(row["liver_feature_key"])
    if key in img_map:
        valid_items.append((key, img_map[key]))

for batch_start in range(0, len(valid_items), BATCH_SIZE):
    batch_items = valid_items[batch_start:batch_start + BATCH_SIZE]
    batch_images = []
    batch_keys = []

    for key, img_path in batch_items:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        batch_images.append(img)
        batch_keys.append(key)

    if not batch_images:
        continue

    input_batch = np.array(batch_images, dtype=np.float32)
    vectors = liver_extractor.predict(input_batch, verbose=0)

    for key, vector in zip(batch_keys, vectors):
        np.save(os.path.join("liver_vectors", f"{key}.npy"), vector.flatten())

print("✅ DONE! Features are ready in /heart_vectors and /liver_vectors")
