import os
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_splits(src_root, out_root, test_size=0.15, val_size=0.15):
    # 1. Gather all file paths and labels
    data = []
    if not os.path.exists(src_root):
        raise FileNotFoundError(f"Source directory '{src_root}' not found.")

    classes = sorted([d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))])
    
    print(f"Found classes: {classes}")
    
    for cls in classes:
        cls_path = os.path.join(src_root, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        print(f" - {cls}: {len(files)} images")
        for f in files:
            data.append({'path': os.path.join(cls_path, f), 'filename': f, 'label': cls})
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        raise ValueError("No images found! Check your source directory path.")

    # 2. Stratified Split (70/15/15)
    # First: Separate Test set (15% of total)
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=42
    )
    
    # Second: Separate Val set from the remaining 85% 
    # To get 15% of the TOTAL, we take (0.15 / 0.85) from the Train+Val pool
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, stratify=train_val_df['label'], random_state=42
    )

    # 3. Copy files to target directories
    split_map = {'train': train_df, 'val': val_df, 'test': test_df}
    
    # Safety: Clean output dir if exists to prevent mixing
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)
    
    for split_name, split_df in split_map.items():
        print(f"\n--- Exporting {split_name} ({len(split_df)} images) ---")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
            target_dir = os.path.join(out_root, split_name, row['label'])
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(row['path'], os.path.join(target_dir, row['filename']))
            
    # 4. Final Verification and CSV Export
    print("\n--- Final Split Counts per Class ---")
    summary = []
    for split in ['train', 'val', 'test']:
        counts = split_map[split]['label'].value_counts().to_dict()
        counts['split'] = split
        summary.append(counts)
    
    summary_df = pd.DataFrame(summary).set_index('split').fillna(0).astype(int)
    print(summary_df)
    
    # Save the splits_summary.csv requested format
    summary_df.to_csv(os.path.join(out_root, "splits_summary.csv"))
    print(f"\nSplits created. Summary saved at {os.path.join(out_root, 'splits_summary.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="UniqueLiverDataset", help="Input clean directory")
    parser.add_argument("--out", default="splits", help="Output directory")
    args = parser.parse_args()
    
    create_splits(args.src, args.out)