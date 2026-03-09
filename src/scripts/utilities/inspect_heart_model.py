import tensorflow as tf
import os

model_path = r'D:\FUSION_TEST\best_heart_model_final.keras'

if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model Name: {model.name}")
        print(f"Input Shape: {model.input_shape}")
        print(f"Output Shape: {model.output_shape}")
        print("\nLayer Summary Table:")
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found at {model_path}")
