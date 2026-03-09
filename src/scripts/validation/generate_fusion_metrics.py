import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, silhouette_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

# --- Configuration ---
DATASET_FILE = r"D:\FUSION_TEST\fusion_master_dataset.npz"
MODELS_DIR = r"D:\FUSION_TEST\models"
METRICS_DIR = r"D:\FUSION_TEST\metrics\fusion"
os.makedirs(METRICS_DIR, exist_ok=True)

print(">>> Generating High-Fidelity Medical Metrics...")

# 1. Load Data and Models
data = np.load(DATASET_FILE)
X_clin = data['clinical']
X_hrt  = data['heart']
X_lvr  = data['liver']
y_true = data['labels']

with open(os.path.join(MODELS_DIR, "clinical_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# Load XGBoost
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model(os.path.join(MODELS_DIR, "xgboost_consultant.json"))

# Load Mediator
fusion_mediator = tf.keras.models.load_model(os.path.join(MODELS_DIR, "fusion_mediator_final.h5"), compile=False)

# 2. Run Inference
X_clin_sc = scaler.transform(X_clin)
xgb_probs = xgb_clf.predict_proba(X_clin_sc)

X_fusion = np.hstack([X_hrt, X_lvr, xgb_probs])
y_probs = fusion_mediator.predict(X_fusion, verbose=0)
y_pred = np.argmax(y_probs, axis=1)

target_names = ["Normal", "Grade I (Mild)", "Grade II (Moderate)", "Grade III (Severe)"]

# 3. Generate Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Fusion CDSS - Confusion Matrix (Total Cohort)')
plt.ylabel('Clinical Ground Truth')
plt.xlabel('AI Mediator Prediction')
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "fusion_confusion_matrix.png"))
plt.close()

# 4. Generate ROC Curves
y_onehot = label_binarize(y_true, classes=[0, 1, 2, 3])
plt.figure(figsize=(10, 8))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'red']
for i, color in enumerate(colors):
    fpr, tpr, _ = roc_curve(y_onehot[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of {target_names[i]} (area = {roc_auc:0.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Fusion CDSS - Multi-Class ROC Analysis (One-vs-Rest)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "fusion_roc_curves.png"))
plt.close()

# 5. Sensitivity vs Specificity Summary Table (Print for Manuscript)
report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
print("\n--- PERFORMANCE SUMMARY ---")
print(f"{'Grade':<20} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10}")
print("-" * 60)
for name in target_names:
    print(f"{name:<20} | {report[name]['f1-score']:>10.4f} | {report[name]['precision']:>10.4f} | {report[name]['recall']:>10.4f}")

# 6. Generate Sensitivity/Specificity Plot
# Sensitivity is Recall
sensitivity = [report[n]['recall'] for n in target_names]
# Specificity for class i: (TN) / (TN + FP)
precision = [report[n]['precision'] for n in target_names]

plt.figure(figsize=(10, 6))
x = np.arange(len(target_names))
plt.bar(x - 0.2, sensitivity, 0.4, label='Sensitivity (Recall)', color='#2ecc71')
plt.bar(x + 0.2, precision, 0.4, label='Precision', color='#3498db')
plt.xticks(x, target_names)
plt.ylabel('Score')
plt.title('Fusion CDSS - Per-Grade Sensitivity and Precision')
plt.legend()
plt.ylim([0, 1.1])
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "fusion_sensitivity_vs_specificity.png"))
plt.close()

print(f"\n[SUCCESS] New metrics generated in {METRICS_DIR}")
