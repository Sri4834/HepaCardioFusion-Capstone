# HepaCardio Fusion CDSS: Multi-Modal Diastolic Dysfunction Grading

This repository contains the complete submission package for the **HepaCardio Fusion CDSS**, a clinical decision support system designed for high-fidelity Diastolic Dysfunction (DD) grading using a Heart-Liver Axis multi-modal approach.

## 🚀 Overview
The system fuses three expert pillars to reach a clinical consensus:
1. **Hepatic Expert**: EfficientNetB0 for METAVIR staging from liver B-mode ultrasound.
2. **Cardiac Expert**: MobileNetV2 + LSTM for EF% estimation from apical 4-chamber echocardiography.
3. **Clinical Consultant**: XGBoost 2.0 for hemodynamic scoring based on ASE 2016 guidelines.

## 📁 Repository Structure
- `docs/`: Includes the full research manuscript (Scopus Q1 standard) and technical blueprint.
- `src/`: Core source code for preprocessing, training, and the CDSS web application.
- `weights/`: Pre-trained models for the expert pillars and the late-fusion mediator.
- `data/`: Real-patient validation gallery with clinical profiles and imaging assets.
- `logs/`: High-resolution metrics, including ROC-AUC curves, Confusion Matrices, and Grad-CAM explainability visuals.
- `configs/`: System hyperparameters and metadata manifest.

## 🛠️ Setup & Reproducibility
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Inference Demo**:
   ```bash
   python src/app_simple.py
   ```
3. **Reproduce Pipeline**:
   Follow the execution protocol detailed in `docs/manuscript.md` (Section 6.2).

## 📄 Manuscript
The central piece of this research, **"Multi-Modal Decision-Level Fusion of Hepatic B-Mode Ultrasound, Cardiac Echo, and Clinical Hemodynamics for Automated Diastolic Dysfunction Grading"**, is located in `docs/manuscript.md`.

## ✨ Key Features
- **Phase 10 Hardened Training**: SMOTE-balanced and Jitter-injected training for clinical robustness.
- **Heart-Liver Axis Integration**: Morphological corroboration of cardiac filling pressures through hepatic texture analysis.
- **Grad-CAM Explainability**: Visual verification of model focus on relevant anatomical landmarks.

---
**Developed by**: Lead Machine Learning Engineering Team | March 2026
