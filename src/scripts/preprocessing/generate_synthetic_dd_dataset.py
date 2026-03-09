import argparse
import os
import numpy as np
import pandas as pd


def _clip_normal(rng: np.random.Generator, mean: float, sd: float, low: float, high: float, size: int) -> np.ndarray:
    values = rng.normal(mean, sd, size)
    return np.clip(values, low, high)


def _sample_ea_ratio(rng: np.random.Generator, grade: str) -> float:
    if grade == "Grade I":
        return float(_clip_normal(rng, mean=0.70, sd=0.10, low=0.40, high=0.79, size=1)[0])
    if grade == "Grade III":
        return float(_clip_normal(rng, mean=2.5, sd=0.40, low=2.00, high=4.00, size=1)[0])
    if grade == "Grade II":
        return float(_clip_normal(rng, mean=1.2, sd=0.25, low=0.81, high=1.99, size=1)[0])
    # Normal
    return float(_clip_normal(rng, mean=1.1, sd=0.2, low=0.8, high=1.5, size=1)[0])


def _sample_lavi(rng: np.random.Generator, underlying: str) -> float:
    if underlying == "Normal":
        return float(_clip_normal(rng, mean=24.0, sd=4.0, low=18.0, high=34.0, size=1)[0])
    elif underlying == "Abnormal":
        return float(_clip_normal(rng, mean=42.0, sd=8.0, low=28.0, high=75.0, size=1)[0])
    return 30.0


def _calculate_official_grade(e_a: float, e_e: float, trv: float, lavi: float, e_vel: float, ef_percent: float) -> tuple[str, int]:
    """Official ASE 2016 Algorithm: Stage 1 (Screening) -> Stage 2 (Grading)"""
    
    # --- STAGE 1: Normal Function vs DD Screening (For Normal EF patients) ---
    c1 = 1 if e_e > 14 else 0
    c2 = 1 if trv > 2.8 else 0
    c3 = 1 if lavi > 34 else 0
    # simplified: using avg e_e instead of septal/lateral for this model
    pts_screening = c1 + c2 + c3 
    
    # If < 50% of criteria are met (0 or 1), it's Normal
    if pts_screening <= 1 and ef_percent >= 50:
        return "Normal", 0
    
    # --- STAGE 2: Grading (For patients with DD or reduced EF) ---
    # Grade III (Restrictive)
    if e_a >= 2.0:
        return "Grade III (Severe)", 3
    
    # Grade I (Mild) vs Grade II (Moderate) Branching
    if e_a <= 0.8 and e_vel <= 50:
        return "Grade I (Mild)", 1
    
    # The "Indeterminate/Moderate" Branch
    # Criteria: E/e' > 14, TRV > 2.8, LAVI > 34 (reuse points)
    if pts_screening >= 2:
        return "Grade II (Moderate)", 2
    else:
        return "Grade I (Mild)", 1


def _calculate_h2fpef_pts(age: float, bmi: float, ee: float, trv: float, afib: bool, htn_meds: bool) -> int:
    pts = 0
    if age > 60: pts += 1
    if bmi > 30: pts += 2
    if ee > 9: pts += 1
    if trv > 2.8: pts += 1
    if afib: pts += 3
    if htn_meds: pts += 1
    return pts


def _load_ef_pool(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        # Generate a realistic distribution of Ejection Fraction (40% to 75%)
        return np.random.normal(58, 8, 1000).clip(40, 75)
    df = pd.read_csv(file_path)
    if "EF" not in df.columns:
        return np.random.normal(58, 8, 1000).clip(40, 75)
    return pd.to_numeric(df["EF"], errors="coerce").dropna().to_numpy()


def _load_video_filenames(videos_dir: str) -> list[str]:
    if not os.path.isdir(videos_dir):
        return [f"video_{i}.avi" for i in range(1000)]
    return [f for f in os.listdir(videos_dir) if f.endswith(".avi")]


def synthesize_dataset(n_rows: int, seed: int, ef_pool: np.ndarray, video_filenames: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    
    stages = ["F0", "F1", "F2", "F3", "F4"]
    stage_probs = [0.25, 0.25, 0.20, 0.18, 0.12]
    
    rows = []
    for i in range(n_rows):
        stage = rng.choice(stages, p=stage_probs)
        
        # Clinical Base
        age = _clip_normal(rng, 55 + stages.index(stage)*2, 10, 20, 90, 1)[0]
        bmi = _clip_normal(rng, 26 + stages.index(stage)*1, 4, 18, 45, 1)[0]
        sbp = _clip_normal(rng, 125 + stages.index(stage)*3, 15, 90, 210, 1)[0]
        dbp = _clip_normal(rng, 80 + stages.index(stage)*1, 10, 50, 120, 1)[0]
        ef_percent = float(rng.choice(ef_pool))
        
        # Stochastic Clinical Flags
        afib = rng.random() < (0.05 + 0.05 * stages.index(stage))
        htn_meds = rng.random() < (0.3 + 0.1 * stages.index(stage))
        
        # Underlying Severity (Probability shifts with Liver Stage)
        # Higher Liver Stage -> Higher chance of underlying DD hemodynamics
        p_normal = max(0.05, 0.4 - 0.08 * stages.index(stage))
        p_grade1 = 0.4
        p_grade2 = 0.1 + 0.05 * stages.index(stage)
        p_grade3 = 1.0 - (p_normal + p_grade1 + p_grade2)
        underlying = rng.choice(["Normal", "Grade I", "Grade II", "Grade III"], p=[p_normal, p_grade1, p_grade2, p_grade3])
        
        # Hemodynamic Parameters based on Underlying Severity
        e_a = _sample_ea_ratio(rng, underlying)
        trv = _clip_normal(rng, 2.2 + 0.3 * ["Normal", "Grade I", "Grade II", "Grade III"].index(underlying), 0.5, 1.8, 4.5, 1)[0]
        e_e = _clip_normal(rng, 8 + 3 * ["Normal", "Grade I", "Grade II", "Grade III"].index(underlying), 3, 5, 28, 1)[0]
        lavi = _sample_lavi(rng, "Abnormal" if underlying != "Normal" else "Normal")
        e_vel = _clip_normal(rng, 60 if underlying != "Grade I" else 45, 15, 30, 130, 1)[0]
        
        # --- PHASE 4 ROBUSTNESS: Calibrated Clinical Ambiguity ---
        # 1. Parameter Jitter (±10% to simulate measurement variability without destroying class bounds)
        age *= rng.uniform(0.95, 1.05)
        bmi *= rng.uniform(0.90, 1.10)
        e_e *= rng.uniform(0.90, 1.10) 
        trv *= rng.uniform(0.90, 1.10)
        e_a *= rng.uniform(0.90, 1.10)
        lavi *= rng.uniform(0.90, 1.10)
        e_vel *= rng.uniform(0.90, 1.10)

        # 2. Expert Deviation (10% Label Noise)
        # 10% chance the 'Ground Truth' label follows the biological intent (underlying)
        # even if noisy measurements suggest otherwise. This simulates occasional expert intuition.
        algo_grade_str, algo_grade_idx = _calculate_official_grade(e_a, e_e, trv, lavi, e_vel, ef_percent)
        
        if rng.random() < 0.10:
            gt_idx = ["Normal", "Grade I", "Grade II", "Grade III"].index(underlying)
        else:
            gt_idx = algo_grade_idx
            
        grade_map = {0: "Normal", 1: "Grade I", 2: "Grade II", 3: "Grade III"}
        grade_idx = gt_idx
        grade_str = grade_map[gt_idx]

        # 30% Pillar Decoupling (Imaging vs Clinical features)
        if rng.random() < 0.30:
            stage = rng.choice(stages)
        
        # H2FPEF calculation
        h2_pts = _calculate_h2fpef_pts(age, bmi, e_e, trv, afib, htn_meds)
        h2_prob = (h2_pts / 9.0) * 100.0
        
        # Fusion-ready risk score
        stage_pts = stages.index(stage) * 1.5
        risk_score = stage_pts + grade_idx * 2.5 + (h2_pts * 0.8)
        
        video_idx = i % len(video_filenames)
        video_filename = video_filenames[video_idx]
        video_stem = os.path.splitext(video_filename)[0]

        rows.append({
            "patient_id": f"PAT_{i:05d}",
            "video_filename": video_filename,
            "heart_feature_key": video_stem,
            "liver_feature_key": f"LIV_{i:05d}",
            "age": round(float(age), 1),
            "bmi": round(float(bmi), 1),
            "systolic_bp": int(sbp),
            "diastolic_bp": int(dbp),
            "ef_percent": round(float(ef_percent), 1),
            "e_e_ratio": round(float(e_e), 2),
            "trv": round(float(trv), 2),
            "e_a_ratio": round(float(e_a), 2),
            "lavi": round(float(lavi), 1),
            "e_velocity": round(float(e_vel), 1),
            "afib": int(afib),
            "htn_meds": int(htn_meds),
            "liver_stage": stage,
            "dd_grade": grade_str,
            "dd_grade_index": grade_idx,
            "h2fpef_score": h2_pts,
            "hfpef_%": round(h2_prob, 2),
            "dd_risk_score": round(risk_score, 2),
            "is_synthetic": 1,
            # Add one-hot style probabilities for mediator
            "risk_prob_normal_pct": 100.0 if grade_idx == 0 else 0.0,
            "risk_prob_grade1_pct": 100.0 if grade_idx == 1 else 0.0,
            "risk_prob_grade2_pct": 100.0 if grade_idx == 2 else 0.0,
            "risk_prob_grade3_pct": 100.0 if grade_idx == 3 else 0.0,
            "dd_risk": 1 if grade_idx > 0 else 0
        })
        
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--out", type=str, default="D:\\FUSION_TEST\\fusion_training_dataset.csv")
    args = parser.parse_args()
    
    ef_pool = _load_ef_pool("D:\\FUSION_TEST\\FileList.csv")
    videos = _load_video_filenames("D:\\FUSION_TEST\\Videos")
    
    df = synthesize_dataset(args.rows, 42, ef_pool, videos)
    df.to_csv(args.out, index=False)
    print(f"Dataset generated: {args.out} ({len(df)} rows)")
    print("Grade Distribution:")
    print(df['dd_grade'].value_counts())

if __name__ == "__main__":
    main()
