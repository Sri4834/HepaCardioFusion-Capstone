"""
Convert Heart Grad-CAM AVI videos to animated GIFs for GitHub markdown inline rendering.
GitHub does NOT support <video> tags inside table cells, but renders animated GIFs natively.
"""
import os
import cv2
import numpy as np
from PIL import Image

INPUT_DIR = r"D:\FUSION_TEST\metrics\heart_gradcam"
OUTPUT_DIR = r"D:\FUSION_TEST\metrics\heart_gradcam"  # Save GIFs alongside AVIs

print("==== Converting AVI Grad-CAM Videos to Animated GIFs ====")

avi_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".avi")])
print(f"Found {len(avi_files)} AVI files to convert.")

for avi_name in avi_files:
    avi_path = os.path.join(INPUT_DIR, avi_name)
    gif_name = avi_name.replace(".avi", ".gif")
    gif_path = os.path.join(OUTPUT_DIR, gif_name)
    
    cap = cv2.VideoCapture(avi_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frame_duration = int(1000 / fps)  # ms per frame for PIL
    
    pil_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to 224px wide to keep GIF size manageable
        h, w = frame.shape[:2]
        new_w = 224
        new_h = int(h * new_w / w)
        frame = cv2.resize(frame, (new_w, new_h))
        # OpenCV BGR → RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(frame_rgb))
    cap.release()
    
    if not pil_frames:
        print(f"  [SKIP] No frames read from {avi_name}")
        continue
    
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration,
        loop=0,     # loop forever
        optimize=True
    )
    
    size_kb = os.path.getsize(gif_path) // 1024
    print(f"  [DONE] {gif_name}  ({len(pil_frames)} frames, {size_kb} KB)")

print("\n==== Conversion Complete ====")
print("GIFs saved to:", OUTPUT_DIR)
