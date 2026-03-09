import os
import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm

# --- Configuration ---
CSV_PATH = r"D:\FUSION_TEST\fusion_training_dataset.csv"
LIVER_POOL_PATH = r"D:\FUSION_TEST\liver_pools\liver_feature_pool_128D.pkl"
HEART_VECTORS_DIR = r"D:\FUSION_TEST\heart_vectors"
OUTPUT_DIR = r"D:\FUSION_TEST"

print("==== Starting Phase 2: Data Synthesis ====")

# Load Clinical Data
df = pd.read_csv(CSV_PATH)
print(f"[1/4] Loaded Clinical Dataset: {len(df)} records")

# Load Liver Features Pool
with open(LIVER_POOL_PATH, 'rb') as f:
    liver_pool = pickle.load(f)

# Ensure pools are not empty
for stage, vectors in liver_pool.items():
    if len(vectors) == 0:
        raise ValueError(f"Liver pool for {stage} is empty!")

print(f"[2/4] Loaded Liver Pools ({', '.join([f'{k}: {len(v)}' for k, v in liver_pool.items()])})")

# Identify continuous clinical features to include in the fusion 
# We'll use robust clinical indicators
clinical_cols = [
    'age', 'bmi', 'systolic_bp', 'diastolic_bp', 
    'ef_percent', 'e_e_ratio', 'trv', 'e_a_ratio', 
    'lavi', 'e_velocity', 'afib', 'htn_meds'
]

X_clinical = []
X_heart = []
X_liver = []
y_labels = []

print("[3/4] Synthesizing 192D+ Dataset instances via Severity Matching...")
skipped = 0

# Random seed for reproducibility
random.seed(42)

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Merging"):
    # 1. Clinical Features
    try:
        clin_feat = row[clinical_cols].values.astype(np.float32)
    except Exception as e:
        skipped += 1
        continue
    
    # 2. Heart Feature (64D)
    video_filename = row['video_filename']
    vector_filename = video_filename.replace('.avi', '.npy')
    heart_path = os.path.join(HEART_VECTORS_DIR, vector_filename)
    
    if os.path.exists(heart_path):
        heart_feat = np.load(heart_path).flatten()
        if heart_feat.shape[0] != 64:
            skipped += 1
            continue
    else:
        skipped += 1
        continue
        
    # 3. Liver Feature (128D) via Monte Carlo Severity Matching
    stage = row['liver_stage']
    if stage not in liver_pool:
        stage = 'F0' # Fallback
        
    liver_feat = random.choice(liver_pool[stage])
    
    # Label: dd_grade_index (0: Normal, 1: Grade I, 2: Grade II, 3: Grade III)
    label = row['dd_grade_index']
    
    X_clinical.append(clin_feat)
    X_heart.append(heart_feat)
    X_liver.append(liver_feat)
    y_labels.append(label)

X_clinical = np.array(X_clinical)
X_heart = np.array(X_heart)
X_liver = np.array(X_liver)
y_labels = np.array(y_labels)

print(f"\n[4/4] Synthesis Complete. Processed {len(y_labels)} records (Skipped: {skipped})")

output_file = os.path.join(OUTPUT_DIR, "fusion_master_dataset.npz")
np.savez(
    output_file, 
    clinical=X_clinical, 
    heart=X_heart, 
    liver=X_liver, 
    labels=y_labels
)

print(f"[SUCCESS] Saved Master Dataset to: {output_file}")
print("-> Matrix Shapes:")
print(f"   Clinical: {X_clinical.shape}")
print(f"   Heart:    {X_heart.shape}")
print(f"   Liver:    {X_liver.shape}")
print(f"   Labels:   {y_labels.shape}")
print("==== Phase 2 Synthesis Complete. Ready for Mediator Training ====")
