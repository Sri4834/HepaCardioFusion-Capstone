import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
import traceback

# --- PATHS ---
HEART_MODEL_PATH = r'D:\FUSION_TEST\best_heart_model_final.keras'
LIVER_MODEL_PATH = r'D:\New folder\training_outputs\liver_model_final.keras'
CSV_PATH = r'D:\DATASET DOWNLOADS\EchoNet-Dynamic 7.4GB avi files\FileList.csv'
VIDEO_DIR = r'D:\DATASET DOWNLOADS\EchoNet-Dynamic 7.4GB avi files\Videos'

def build_heart_model():
    img_shape = (112, 112, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights=None)
    inputs = layers.Input(shape=(16, 112, 112, 3))
    x = layers.TimeDistributed(base_model)(inputs)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.LSTM(64, return_sequences=False, name="temporal_memory")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='linear', name="ef_prediction")(x)
    return models.Model(inputs, outputs)

def verify():
    print("--- STARTING COMPREHENSIVE AUDIT & VERIFICATION ---")
    
    # 1. Heart Model Test
    print("\n[1/2] Heart Model Verification...")
    try:
        h_model = build_heart_model()
        h_model.load_weights(HEART_MODEL_PATH)
        print("SUCCESS: Heart Model architecture reconstructed and weights loaded.")
    except Exception as e:
        print(f"FAILED: Heart Model reconstruction failed: {e}")

    # 2. Liver Model Test (User's Concern)
    print("\n[2/2] Liver Model Verification...")
    if os.path.exists(LIVER_MODEL_PATH):
        try:
            l_model = keras.models.load_model(LIVER_MODEL_PATH, compile=False)
            print("SUCCESS: Liver model loaded successfully with modern Keras.")
            print("Liver Model Input Shape:", l_model.input_shape)
        except Exception as e:
            print(f"FAILED: Liver model load failed: {e}")
            traceback.print_exc()
    else:
        print(f"SKIP: Liver model file not found at {LIVER_MODEL_PATH}")

    print("\n--- VERIFICATION COMPLETE ---")

if __name__ == "__main__":
    verify()
