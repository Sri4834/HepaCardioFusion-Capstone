import os
import json
import hashlib
import itertools
import numpy as np
import pandas as pd
import cv2
import app

liver_root = "D:/DATASET DOWNLOADS/STRATIFIED UNIQUE DATASET FOR MODEL EVALUATION-physical_holdout_test {Gpt's said me that these are seperated fiels but I dont believe them because my current liver model gives 99+ confidence for these or any stagelabeldimage}"
heart_dir = "C:/Users/dshv1/Videos/Screen Recordings/AVI DD'S"

out_cases = "D:/FUSION_TEST/clinical_testcases_generated.csv"
out_results = "D:/FUSION_TEST/clinical_testcase_fusion_results.csv"
out_dup_csv = "D:/FUSION_TEST/liver_duplicate_audit.csv"
out_summary = "D:/FUSION_TEST/clinical_and_duplicate_summary.txt"

stage_labels = app.LIVER_STAGE_LABELS
grade_labels = app.GRADE_LABELS

# Clinically inspired test cases (ASE/EACVI-style trends for DD patterning)
cases = [
    {"case_id":"C1_normal_pattern", "age":34, "bmi":22, "systolic_bp":116, "diastolic_bp":74, "e_e_ratio":7.5, "trv":2.2, "e_a_ratio":1.3},
    {"case_id":"C2_borderline_relax", "age":47, "bmi":26, "systolic_bp":126, "diastolic_bp":80, "e_e_ratio":10.5, "trv":2.4, "e_a_ratio":0.9},
    {"case_id":"C3_grade1_like", "age":55, "bmi":28, "systolic_bp":134, "diastolic_bp":84, "e_e_ratio":12.0, "trv":2.5, "e_a_ratio":0.8},
    {"case_id":"C4_grade2_like", "age":61, "bmi":30, "systolic_bp":142, "diastolic_bp":86, "e_e_ratio":15.5, "trv":2.9, "e_a_ratio":1.5},
    {"case_id":"C5_grade3_like", "age":69, "bmi":33, "systolic_bp":154, "diastolic_bp":92, "e_e_ratio":18.5, "trv":3.3, "e_a_ratio":2.1},
    {"case_id":"C6_high_bp_low_ee", "age":58, "bmi":31, "systolic_bp":160, "diastolic_bp":95, "e_e_ratio":9.5, "trv":2.4, "e_a_ratio":1.0},
    {"case_id":"C7_obese_intermediate", "age":63, "bmi":35, "systolic_bp":146, "diastolic_bp":88, "e_e_ratio":14.0, "trv":2.7, "e_a_ratio":1.2},
    {"case_id":"C8_elderly_high_risk", "age":74, "bmi":29, "systolic_bp":150, "diastolic_bp":90, "e_e_ratio":19.0, "trv":3.4, "e_a_ratio":2.0},
]

pd.DataFrame(cases).to_csv(out_cases, index=False)

# Pick one real heart video + one real liver image from each stage
heart_files = [os.path.join(heart_dir, f) for f in os.listdir(heart_dir) if f.lower().endswith('.avi')] if os.path.isdir(heart_dir) else []
heart_files.sort()
selected_heart = heart_files[0] if heart_files else None

stage_image = {}
if os.path.isdir(liver_root):
    for st in stage_labels:
        d = os.path.join(liver_root, st)
        if not os.path.isdir(d):
            continue
        picked = None
        for root, _, files in os.walk(d):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in {'.png', '.jpg', '.jpeg'}:
                    picked = os.path.join(root, fn)
                    break
            if picked:
                break
        stage_image[st] = picked

# Run testcase matrix: each clinical case x each available stage image x one heart video
rows = []
if selected_heart and len(stage_image) > 0:
    h_out = app.extract_heart_outputs(selected_heart, app.heart_full)
    for c in cases:
        for st, img_path in stage_image.items():
            if not img_path:
                continue
            try:
                l_out = app.extract_liver_outputs(img_path, app.liver_full)
                form = {
                    'age': str(c['age']),
                    'bmi': str(c['bmi']),
                    'systolic_bp': str(c['systolic_bp']),
                    'diastolic_bp': str(c['diastolic_bp']),
                    'e_e_ratio': str(c['e_e_ratio']),
                    'trv': str(c['trv']),
                    'e_a_ratio': str(c['e_a_ratio']),
                }
                c_vec, _ = app.parse_clinical(form, h_out['ef_percent'], l_out['stage_index'])
                clin_latent = app.extract_clinical_latent(c_vec)
                h_vec, l_vec, meta = app.build_fusion_vectors(h_out, l_out, c_vec, clin_latent, app.FUSION_INPUT_MODE)
                primary = app.run_keras_inference(h_vec, l_vec, c_vec)
                use_fb, reason = app.should_fallback(primary)
                final = app.run_torch_inference(h_vec, l_vec, c_vec) if use_fb else primary
                pred_idx = int(final['pred_idx'])
                rows.append({
                    'case_id': c['case_id'],
                    'heart_video': os.path.basename(selected_heart),
                    'liver_image': os.path.basename(img_path),
                    'liver_stage_from_model': int(l_out['stage_index']),
                    'predicted_dd_grade': grade_labels[pred_idx],
                    'predicted_dd_idx': pred_idx,
                    'fusion_confidence': float(final['confidence']),
                    'used_fallback': bool(use_fb),
                    'fallback_reason': reason,
                    'clinical_latent_used': bool(meta.get('clinical_latent_used', False)),
                })
            except Exception as e:
                rows.append({
                    'case_id': c['case_id'],
                    'heart_video': os.path.basename(selected_heart),
                    'liver_image': os.path.basename(img_path),
                    'liver_stage_from_model': -1,
                    'predicted_dd_grade': 'ERROR',
                    'predicted_dd_idx': -1,
                    'fusion_confidence': 0.0,
                    'used_fallback': False,
                    'fallback_reason': str(e),
                    'clinical_latent_used': False,
                })

results_df = pd.DataFrame(rows)
results_df.to_csv(out_results, index=False)

# Duplicate audit: exact hash + perceptual dHash similarity for same-patient-like duplicates
img_paths = []
if os.path.isdir(liver_root):
    for root, _, files in os.walk(liver_root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in {'.png', '.jpg', '.jpeg'}:
                img_paths.append(os.path.join(root, fn))


def exact_sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dhash64(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = img[:, 1:] > img[:, :-1]
    bits = ''.join('1' if v else '0' for v in diff.flatten())
    return int(bits, 2)


def hamming(a, b):
    return (a ^ b).bit_count()

exact_map = {}
phash_rows = []
for p in img_paths:
    try:
        sh = exact_sha1(p)
        exact_map.setdefault(sh, []).append(p)
    except Exception:
        pass
    dh = dhash64(p)
    if dh is not None:
        phash_rows.append((p, dh))

exact_dup_groups = [v for v in exact_map.values() if len(v) > 1]

near_dups = []
# Conservative threshold: distance <= 4 suggests very similar frames/scans
for (p1, h1), (p2, h2) in itertools.combinations(phash_rows, 2):
    d = hamming(h1, h2)
    if d <= 4:
        near_dups.append((p1, p2, d))

dup_records = []
for gidx, g in enumerate(exact_dup_groups, start=1):
    for p in g:
        dup_records.append({'dup_type': 'exact', 'group_id': gidx, 'path_a': p, 'path_b': '', 'distance': 0})

base_gid = len(exact_dup_groups)
for i, (a, b, d) in enumerate(near_dups, start=1):
    dup_records.append({'dup_type': 'near', 'group_id': base_gid + i, 'path_a': a, 'path_b': b, 'distance': int(d)})

pd.DataFrame(dup_records).to_csv(out_dup_csv, index=False)

summary = {
    'clinical_testcases': {
        'generated_cases': len(cases),
        'selected_heart_video': os.path.basename(selected_heart) if selected_heart else None,
        'available_liver_stage_images': {k: (os.path.basename(v) if v else None) for k, v in stage_image.items()},
        'result_rows': int(len(results_df)),
        'outputs': {
            'cases_csv': out_cases,
            'results_csv': out_results,
        },
    },
    'liver_duplicate_audit': {
        'total_images_scanned': len(img_paths),
        'exact_duplicate_groups': len(exact_dup_groups),
        'exact_duplicate_files': int(sum(len(g) for g in exact_dup_groups)),
        'near_duplicate_pairs_dhash_le_4': len(near_dups),
        'audit_csv': out_dup_csv,
        'note': 'Near-duplicates indicate scan/frame similarity; true same-patient identity cannot be confirmed without patient IDs/metadata.'
    }
}

with open(out_summary, 'w', encoding='utf-8') as f:
    f.write(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))
print('SUMMARY_FILE=' + out_summary)
