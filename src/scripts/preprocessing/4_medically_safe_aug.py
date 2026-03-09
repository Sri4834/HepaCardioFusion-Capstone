import os, cv2, random, argparse
import albumentations as A
import pandas as pd
from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument("--src", default="splits/train")
p.add_argument("--out", default="aug_train")
p.add_argument("--target_count", type=int, default=1000) # per class
args = p.parse_args()

# Medically safe transforms using Albumentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=0, p=0.3),
])

for cls in os.listdir(args.src):
    cls_path = os.path.join(args.src, cls)
    out_path = os.path.join(args.out, cls)
    os.makedirs(out_path, exist_ok=True)
    
    files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    if not files: continue
    
    # 1. Copy original reps first
    for f in files:
        img = cv2.imread(os.path.join(cls_path, f))
        cv2.imwrite(os.path.join(out_path, f), img)
    
    # 2. Augment up to target_count
    num_to_make = args.target_count - len(files)
    print(f"Augmenting Class {cls}: Need {num_to_make} more images...")
    
    for i in tqdm(range(num_to_make)):
        ref_file = random.choice(files)
        image = cv2.imread(os.path.join(cls_path, ref_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = transform(image=image)['image']
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(out_path, f"aug_{i}_{ref_file}"), augmented)

print(f"Offline augmentation complete. New training set at: {args.out}")
