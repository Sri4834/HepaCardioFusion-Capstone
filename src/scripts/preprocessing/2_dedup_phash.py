# 1_dedup_phash.py
import os
import argparse
import shutil
import pandas as pd
from PIL import Image
import imagehash
from collections import defaultdict, Counter
from tqdm import tqdm

def compute_hashes(src_root, hash_size=16):
    """Computes pHash for all images in the source directory."""
    data = []
    # Collect all image paths and labels from fan_crops
    for cls in sorted(os.listdir(src_root)):
        cls_path = os.path.join(src_root, cls)
        if not os.path.isdir(cls_path):
            continue
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        for fn in os.listdir(cls_path):
            if fn.lower().endswith(valid_exts):
                data.append({
                    "path": os.path.join(cls_path, fn),
                    "filename": fn,
                    "label": cls
                })

    hash_to_files = defaultdict(list)
    print(f"--- Computing phashes (Size: {hash_size}x{hash_size}) for {len(data)} images ---")
    
    for item in tqdm(data, desc="pHash Extraction"):
        try:
            with Image.open(item["path"]) as img:
                # Convert to grayscale for consistent hashing
                h = imagehash.phash(img, hash_size=hash_size)
                hash_to_files[str(h)].append(item)
        except Exception as e:
            print(f"\n[ERROR] Failed to process {item['path']}: {e}")
            
    return hash_to_files

def cluster_hashes(hash_map, hamming_thresh=2):
    """Clusters hashes based on Hamming distance."""
    groups = []
    keys = list(hash_map.keys())
    seen = set()

    print(f"--- Clustering near-duplicates (Hamming Thresh: {hamming_thresh}) ---")
    for i, k1 in enumerate(tqdm(keys, desc="Clustering")):
        if k1 in seen:
            continue
        
        group_keys = [k1]
        h1 = imagehash.hex_to_hash(k1)
        seen.add(k1)

        for k2 in keys[i+1:]:
            if k2 in seen:
                continue
            h2 = imagehash.hex_to_hash(k2)
            if h1 - h2 <= hamming_thresh:
                group_keys.append(k2)
                seen.add(k2)
        
        members = []
        for kh in group_keys:
            members.extend(hash_map[kh])
        groups.append(members)
        
    return groups

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Data Deduplication")
    parser.add_argument("--src", default="fan_crops", help="Input directory")
    parser.add_argument("--out", default="UniqueLiverDataset", help="Output directory")
    parser.add_argument("--hash_size", type=int, default=16)
    parser.add_argument("--thresh", type=int, default=2) # Strict threshold as requested
    args = parser.parse_args()

    # 1. Pipeline Execution
    hash_map = compute_hashes(args.src, hash_size=args.hash_size)
    groups = cluster_hashes(hash_map, hamming_thresh=args.thresh)
    
    os.makedirs(args.out, exist_ok=True)
    rows = []
    conflicts_found = 0

    print(f"--- Exporting Unique Images to {args.out} ---")
    for gid, members in enumerate(tqdm(groups, desc="Exporting")):
        labels = [m["label"] for m in members]
        label_counts = Counter(labels)
        maj_label = label_counts.most_common(1)[0][0]
        is_conflict = len(label_counts) > 1
        
        if is_conflict:
            conflicts_found += 1
            # Log specific conflict to console for your review
            conflict_details = ", ".join([f"{lbl}({count})" for lbl, count in label_counts.items()])
            print(f"\n[CONFLICT WARNING] Group {gid}: Labels found: {conflict_details}")
            for m in members:
                print(f"  -> {m['path']}")

        # Select representative (best quality/integrity)
        best = max(members, key=lambda m: os.path.getsize(m["path"]))
        
        # Prepare for CSV reporting
        for m in members:
            rows.append({
                "group_id": gid,
                "original_path": m["path"],
                "original_label": m["label"],
                "assigned_label": maj_label,
                "is_representative": int(m["path"] == best["path"]),
                "conflict_group": int(is_conflict)
            })

        # Copy the representative to the unique dataset
        out_cls_dir = os.path.join(args.out, maj_label)
        os.makedirs(out_cls_dir, exist_ok=True)
        dest = os.path.join(out_cls_dir, best["filename"])
        
        # In case of filename collisions across classes, append group ID
        if os.path.exists(dest):
            name, ext = os.path.splitext(best["filename"])
            dest = os.path.join(out_cls_dir, f"{name}_g{gid}{ext}")
            
        shutil.copy2(best["path"], dest)

    # 2. Final Reporting
    df = pd.DataFrame(rows)
    df.to_csv("dedupe_mapping.csv", index=False)
    
    print("\n" + "="*40)
    print("DEDUPLICATION SUMMARY")
    print("="*40)
    print(f"Total Images Analyzed:   {len(df)}")
    print(f"Unique Images Extracted: {len(groups)}")
    print(f"Duplicates Removed:      {len(df) - len(groups)}")
    print(f"Label Conflicts Found:   {conflicts_found}")
    print(f"Mapping saved to:        dedupe_mapping.csv")
    print("="*40)
    if conflicts_found > 0:
        print("[ACTION REQUIRED] Please audit 'dedupe_mapping.csv' where 'conflict_group' == 1.")
