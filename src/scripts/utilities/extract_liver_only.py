import os
import cv2
import numpy as np
import pandas as pd
import keras
from tensorflow.keras.models import load_model, Model


LIVER_DIR = "Dataset"
CSV_PATH = "fusion_training_dataset.csv"
LIVER_MODEL_PATH = "best_vit_model.keras"
LIVER_FEATURE_LAYER = "ExtractToken"
BATCH_SIZE = 32

os.makedirs("liver_vectors", exist_ok=True)


def _normalize_stage(value: str) -> str:
    return str(value).strip().upper()


def build_stage_pools(liver_dir: str):
    stage_pool = {f"F{i}": [] for i in range(5)}
    valid_ext = (".png", ".jpg", ".jpeg")

    for stage in stage_pool.keys():
        stage_path = os.path.join(liver_dir, stage)
        if os.path.isdir(stage_path):
            files = [
                os.path.join(stage_path, f)
                for f in os.listdir(stage_path)
                if f.lower().endswith(valid_ext)
            ]
            files.sort()
            stage_pool[stage] = files

    return stage_pool


def build_worklist(df: pd.DataFrame, stage_pool: dict):
    stage_counters = {k: 0 for k in stage_pool.keys()}
    worklist = []
    skipped_rows = 0

    for _, row in df.iterrows():
        key = str(row.get("liver_feature_key", "")).strip()
        stage = _normalize_stage(row.get("liver_stage", ""))

        if not key or stage not in stage_pool or not stage_pool[stage]:
            skipped_rows += 1
            continue

        pool = stage_pool[stage]
        idx = stage_counters[stage] % len(pool)
        real_path = pool[idx]
        stage_counters[stage] += 1
        worklist.append((key, real_path))

    return worklist, skipped_rows


print("🚀 Loading ViT liver model...")
keras.config.enable_unsafe_deserialization()
liver_full = load_model(LIVER_MODEL_PATH, safe_mode=False)
liver_extractor = Model(
    inputs=liver_full.input,
    outputs=liver_full.get_layer(LIVER_FEATURE_LAYER).output,
)

print("🔍 Building stage-aware mapping...")
df = pd.read_csv(CSV_PATH)
stage_pool = build_stage_pools(LIVER_DIR)
worklist, skipped = build_worklist(df, stage_pool)

for stage in sorted(stage_pool.keys()):
    print(f"{stage} pool size: {len(stage_pool[stage])}")
print(f"✅ Mappings ready: {len(worklist)} rows | skipped: {skipped}")

print("📸 Extracting liver vectors (768D)...")
written = 0
for start in range(0, len(worklist), BATCH_SIZE):
    batch = worklist[start:start + BATCH_SIZE]
    batch_imgs = []
    batch_keys = []

    for key, img_path in batch:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        batch_imgs.append(img)
        batch_keys.append(key)

    if not batch_imgs:
        continue

    preds = liver_extractor.predict(np.array(batch_imgs, dtype=np.float32), verbose=0)

    for key, vec in zip(batch_keys, preds):
        np.save(os.path.join("liver_vectors", f"{key}.npy"), vec.flatten())
        written += 1

    if (start // BATCH_SIZE) % 10 == 0:
        print(f"Progress: {min(start + BATCH_SIZE, len(worklist))}/{len(worklist)} | saved: {written}")

print(f"✅ DONE! Liver vectors saved: {written}")
