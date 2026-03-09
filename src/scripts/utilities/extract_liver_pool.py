import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pickle

# --- Environment Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
LIVER_MODELS_DIR = r"D:\New folder\training_outputs"
LIVER_MODELS_OPT_DIR = r"D:\New folder\training_outputs_opt"
DATASET_DIR = r"D:\New folder\UniqueLiverDataset"
OUTPUT_DIR = r"D:\FUSION_TEST\liver_pools"
CLASSES = ['F0', 'F1', 'F2', 'F3', 'F4']
IMG_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("==== Starting Phase 1: Liver Feature Pool Extraction ====")

def build_feature_extractor(img_size=224, is_optimized=False):
    """Rebuild the EfficientNetB0 architecture and truncate at the 128D Dense layer"""
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=(img_size, img_size, 3)
    )
    
    unfreeze_layers = 40 if is_optimized else 20
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_layers]: layer.trainable = False

    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # 128D Feature Layer
    features = tf.keras.layers.Dense(128, activation='relu', name='feature_dense')(x)
    
    x = tf.keras.layers.Dropout(0.3)(features)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    
    full_model = tf.keras.models.Model(inputs, outputs)
    
    # The actual feature extractor outputs the 128D layer
    feature_extractor = tf.keras.models.Model(inputs=full_model.input, outputs=features)
    return full_model, feature_extractor

print("[1/4] Building and Loading Hybrid Liver Engines...")

baseline_full, baseline_extractor = build_feature_extractor(is_optimized=False)
w_baseline = os.path.join(LIVER_MODELS_DIR, "liver_model_final.keras")

# Keras 3 struggles with .keras files that are actually HDF5 weights. We copy it to .h5.
import shutil
w_h5 = w_baseline.replace('.keras', '.h5')
if not os.path.exists(w_h5):
    print(f" -> Converting legacy model extension to .h5...")
    shutil.copyfile(w_baseline, w_h5)

try:
    baseline_full.load_weights(w_h5)
    print(f" -> Baseline Engine loaded ({w_h5})")
except Exception as e:
    print(f"Failed to load weights: {e}. Trying legacy h5 load...")
    # Attempt legacy load if needed, but standard should work
    baseline_full.load_weights(w_h5)


# We will use the baseline engine mainly to extract consistent 128D features.
# For simplicity, we just use baseline features, but you could concatenate if needed.
# Since the goal is 192D (64D + 128D), we will strictly use the 128D Baseline vectors.

liver_pool = {cls: [] for cls in CLASSES}

print("\n[2/4] Scanning UniqueLiverDataset...")
for cls in CLASSES:
    cls_path = os.path.join(DATASET_DIR, cls)
    if not os.path.exists(cls_path):
        print(f"Warning: {cls_path} not found.")
        continue
    
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f" -> Class {cls}: Found {len(images)} images.")
    
    for img_name in tqdm(images, desc=f"Extracting {cls}", leave=False):
        img_path = os.path.join(cls_path, img_name)
        try:
            img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Batch size = 1
            
            # Extract 128D feature
            features = baseline_extractor.predict(img_array, verbose=0)[0]
            liver_pool[cls].append(features)
        except Exception as e:
            print(f"\nError processing {img_name}: {e}")

print("\n[3/4] Validating Feature Pools...")
for cls in CLASSES:
    pool_size = len(liver_pool[cls])
    print(f" -> {cls} Pool: {pool_size} vectors (Shape: {(pool_size, 128) if pool_size > 0 else 'EMPTY'})")

print("\n[4/4] Saving Pools to Disk...")
output_file = os.path.join(OUTPUT_DIR, "liver_feature_pool_128D.pkl")
with open(output_file, 'wb') as f:
    pickle.dump(liver_pool, f)

print(f"\n[SUCCESS] Liver Feature Pool saved to: {output_file}")
print("==== Phase 1 Complete. Ready for Synthesis ====")
