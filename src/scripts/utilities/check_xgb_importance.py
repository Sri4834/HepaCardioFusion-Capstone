import xgboost as xgb
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = r"D:\FUSION_TEST\models"
xgb_model_path = os.path.join(OUTPUT_DIR, "xgboost_consultant.json")

# Load model
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model(xgb_model_path)

clinical_cols = [
    'age', 'bmi', 'systolic_bp', 'diastolic_bp', 
    'ef%', 'e_e_ratio', 'trv', 'e_a_ratio', 
    'dd_risk_score', 'hfpef_%'
]

# Get feature importances
importances = xgb_clf.feature_importances_

print("=== XGBoost Feature Importances ===")
for col, imp in zip(clinical_cols, importances):
    print(f"{col}: {imp:.4f}")

