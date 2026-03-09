import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import xgboost as xgb
import pickle
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- MODEL PATHS ---
LIVER_MODEL_PATH = r"D:\New folder\submission_package\weights\liver_model_primary.keras"
HEART_MODEL_PATH = r"D:\FUSION_TEST\best_heart_model_final.keras"
XGB_MODEL_PATH   = r"D:\FUSION_TEST\models\xgboost_consultant.json"
SCALER_PATH      = r"D:\FUSION_TEST\models\clinical_scaler.pkl"
FUSION_MODEL_PATH= r"D:\FUSION_TEST\models\fusion_mediator_final.h5"

print(">>> Initializing Clinical Decision Support System (CDSS)...")

# 1. Load Liver Model (exact architecture from New Folder project)
print("[1/5] Loading Liver Feature Extractor...")
import tf_keras
_liver_base = tf_keras.applications.EfficientNetB0(
    include_top=False, weights='imagenet', input_shape=(224,224,3)
)
_liver_base.trainable = True
for layer in _liver_base.layers[:-20]: layer.trainable = False
_liver_inp = tf_keras.layers.Input(shape=(224,224,3))
_lx = _liver_base(_liver_inp, training=False)
_lx = tf_keras.layers.GlobalAveragePooling2D()(_lx)
_lx = tf_keras.layers.BatchNormalization()(_lx)
_lx = tf_keras.layers.Dropout(0.5)(_lx)
_liver_feat_layer = tf_keras.layers.Dense(128, activation='relu')(_lx)
_lx = tf_keras.layers.Dropout(0.3)(_liver_feat_layer)
_liver_out = tf_keras.layers.Dense(5, activation='softmax')(_lx)
liver_model_full = tf_keras.Model(_liver_inp, _liver_out)
liver_model_full.load_weights(r"D:\FUSION_TEST\liver_models\liver_model_baseline.keras", by_name=True)
# Extract the 128D feature layer (dense layer before final dropout)
for _li, _ll in enumerate(liver_model_full.layers):
    if hasattr(_ll, 'units') and _ll.units == 128:
        _feat_idx = _li
liver_feature_extractor = tf_keras.Model(inputs=liver_model_full.inputs, outputs=liver_model_full.layers[_feat_idx].output)
print("    ✓ Liver feature extractor ready (128D)")


# 2. Load Heart Model
print("[2/5] Loading Heart Feature Extractor...")
def build_heart_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(112, 112, 3), include_top=False, weights=None)
    inputs = tf.keras.layers.Input(shape=(16, 112, 112, 3))
    x = tf.keras.layers.TimeDistributed(base_model)(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False, name="temporal_memory")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear', name="ef_prediction")(x)
    return tf.keras.models.Model(inputs, outputs)

heart_model_full = build_heart_model()
heart_model_full.load_weights(HEART_MODEL_PATH)
heart_feature_extractor = tf.keras.Model(inputs=heart_model_full.inputs, outputs=heart_model_full.get_layer('temporal_memory').output)

# 3. Load Clinical XGBoost & Scaler
print("[3/5] Loading XGBoost Clinical Consultant...")
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model(XGB_MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    clinical_scaler = pickle.load(f)

# 4. Load Fusion Mediator
print("[4/5] Loading Fusion Mediator...")
try:
    fusion_mediator = tf.keras.models.load_model(FUSION_MODEL_PATH, compile=False)
except:
    fusion_mediator = tf_keras.models.load_model(FUSION_MODEL_PATH, compile=False)

print("\n[SUCCESS] All Models Loaded Successfully! Server starting on port 5000.")

def preprocess_liver(img_path):
    img = cv2.imread(img_path)
    # Fan crop placeholder (using basic resize for speed in testing)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return np.expand_dims(img, axis=0)

def preprocess_heart(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, 16, dtype=int)
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if idx in indices:
            # Revert to standard resize (stretch) as used in training
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame / 255.0)
        if len(frames) == 16: break
    cap.release()
    while len(frames) < 16: 
        if len(frames) > 0: frames.append(frames[-1])
        else: frames.append(np.zeros((112, 112, 3)))
    return np.expand_dims(np.array(frames), axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Save uploaded files
        liver_file = request.files.get('liver_image')
        heart_file = request.files.get('heart_video')
        
        liver_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_liver.jpg")
        heart_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_heart.avi")
        
        liver_file.save(liver_path)
        heart_file.save(heart_path)
        
        # 2. Extract Features
        liver_tensor = preprocess_liver(liver_path)
        # 2a. Predict Liver Stage for scoring
        liver_preds = liver_model_full.predict(liver_tensor, verbose=0)[0]
        pred_stage_idx = np.argmax(liver_preds)
        stage_map = {0: "F0", 1: "F1", 2: "F2", 3: "F3", 4: "F4"}
        pred_stage = stage_map.get(pred_stage_idx, "F0")
        liv_feats = liver_feature_extractor.predict(liver_tensor, verbose=0) 
        
        heart_tensor = preprocess_heart(heart_path)
        # 2b. Use FULL heart model for AI-calculated EF%
        heart_raw_pred = heart_model_full.predict(heart_tensor, verbose=0)
        predicted_ef = float(heart_raw_pred[0][0])
        hrt_feats = heart_feature_extractor.predict(heart_tensor, verbose=0) 
        
        # 3. Dynamic Clinical Pillar Calculation
        age = float(request.form.get('age', 55))
        bmi = float(request.form.get('bmi', 28))
        sbp = float(request.form.get('systolic', 130))
        dbp = float(request.form.get('diastolic', 85))
        ee = float(request.form.get('ee_ratio', 10))
        trv = float(request.form.get('trv', 2.5))
        ea = float(request.form.get('ea_ratio', 1.0))
        lavi = float(request.form.get('lavi', 28.0))
        e_vel = float(request.form.get('e_velocity', 65.0))
        afib = int(request.form.get('afib', 0))
        htn_meds = int(request.form.get('htn_meds', 0))
        
        # 3a. Official ASE 2016 Algorithm Mapping for scoring
        # (Mirroring generate_synthetic_dd_dataset.py logic)
        c1 = 1 if ee > 14 else 0
        c2 = 1 if trv > 2.8 else 0
        c3 = 1 if lavi > 34 else 0
        pts_screening = c1 + c2 + c3
        
        if pts_screening <= 1 and predicted_ef >= 50:
            grade_idx_est = 0
        elif ea >= 2.0:
            grade_idx_est = 3
        elif ea <= 0.8 and e_vel <= 50:
            grade_idx_est = 1
        else:
            grade_idx_est = 2 if pts_screening >= 2 else 1
            
        # 3b. H2FPEF official calculation
        h2_pts = 0
        if age > 60: h2_pts += 1
        if bmi > 30: h2_pts += 2
        if ee > 9: h2_pts += 1
        if trv > 2.8: h2_pts += 1
        if afib == 1: h2_pts += 3
        if htn_meds == 1: h2_pts += 1
        h2_prob = (h2_pts / 9.0) * 100.0
        
        # 3c. DD-Risk Score
        stage_pts = {0:0.0, 1:0.5, 2:1.0, 3:2.0, 4:3.0}[pred_stage_idx]
        dd_risk_score = stage_pts + (grade_idx_est * 1.5) + (h2_pts * 0.5)

        # Formulate 15D Clinical Vector for XGBoost
        # Order: age, bmi, sbp, dbp, ef, ee, trv, ea, lavi, e_vel, afib, htn, h2_pts, h2_prob, risk
        clin_data = [
            age, bmi, sbp, dbp, predicted_ef, 
            ee, trv, ea, lavi, e_vel, 
            afib, htn_meds, float(h2_pts), h2_prob, dd_risk_score
        ]
        
        clin_arr = np.array(clin_data).reshape(1, -1)
        clin_scaled = clinical_scaler.transform(clin_arr)
        
        xgb_risk_probs = xgb_clf.predict_proba(clin_scaled) 
        
        # 4. Fusion Mediator Inference
        fusion_vector = np.hstack([hrt_feats, liv_feats, xgb_risk_probs]) 
        
        final_probs = fusion_mediator.predict(fusion_vector, verbose=0)[0]
        pred_idx = np.argmax(final_probs)
        
        grades = ["Normal", "Grade I (Mild)", "Grade II (Moderate)", "Grade III (Severe)"]
        
        print(f">>> DEBUG: AI EF%={predicted_ef:.1f} | Stage={pred_stage} | RiskScore={dd_risk_score:.2f} | Final={grades[pred_idx]}")

        response = {
            'status': 'success',
            'grade': grades[pred_idx],
            'confidence': float(final_probs[pred_idx] * 100),
            'ai_ef': round(predicted_ef, 2),
            'ai_stage': pred_stage,
            'h2fpef_score': h2_pts,
            'h2fpef_prob': round(h2_prob, 2),
            'xgb_risk': [float(p) for p in xgb_risk_probs[0]],
            'all_probs': [float(p) for p in final_probs]
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f">>> ERROR: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
