# train_fusion_dls_keras.py
import argparse
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


CLINICAL_COLUMNS = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "ef%",
    "e_e_ratio",
    "trv",
    "e_a_ratio",
    "liver_stage_idx",
]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


class ClinicalScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, df: pd.DataFrame):
        values = df[CLINICAL_COLUMNS].to_numpy(dtype=np.float32)
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        values = df[CLINICAL_COLUMNS].to_numpy(dtype=np.float32)
        return (values - self.mean) / self.std

    def transform_row(self, row: pd.Series) -> np.ndarray:
        values = row[CLINICAL_COLUMNS].to_numpy(dtype=np.float32)
        return (values - self.mean) / self.std


def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    if "liver_stage_idx" not in df.columns:
        if "liver_stage" not in df.columns:
            raise ValueError("CSV must include either 'liver_stage_idx' or 'liver_stage'.")
        df["liver_stage_idx"] = (
            df["liver_stage"].astype(str).str.extract(r"(\d)").astype(int)
        )

    required = [
        "heart_feature_key",
        "liver_feature_key",
        "hfpef_%",
        "dd_grade_index",
        *CLINICAL_COLUMNS,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["heart_feature_key"] = df["heart_feature_key"].astype(str).str.strip()
    df["liver_feature_key"] = df["liver_feature_key"].astype(str).str.strip()
    return df


def load_vectors_and_targets(
    df: pd.DataFrame,
    heart_dir: str,
    liver_dir: str,
    scaler: ClinicalScaler,
):
    n = len(df)
    heart = np.zeros((n, 64), dtype=np.float32)
    liver = np.zeros((n, 128), dtype=np.float32)
    clin = np.zeros((n, 9), dtype=np.float32)
    hfpef = np.zeros((n,), dtype=np.float32)
    grade = np.zeros((n,), dtype=np.int32)

    for i, (_, row) in enumerate(df.iterrows()):
        h_path = os.path.join(heart_dir, f"{row['heart_feature_key']}.npy")
        l_path = os.path.join(liver_dir, f"{row['liver_feature_key']}.npy")

        if not os.path.exists(h_path):
            raise FileNotFoundError(f"Missing heart vector: {h_path}")
        if not os.path.exists(l_path):
            raise FileNotFoundError(f"Missing liver vector: {l_path}")

        h_vec = np.load(h_path).astype(np.float32).reshape(-1)
        l_vec = np.load(l_path).astype(np.float32).reshape(-1)

        if h_vec.shape[0] != 64:
            raise ValueError(f"Heart vector must be 64D, got {h_vec.shape[0]} at {h_path}")
        if l_vec.shape[0] != 128:
            raise ValueError(f"Liver vector must be 128D, got {l_vec.shape[0]} at {l_path}")

        heart[i] = h_vec
        liver[i] = l_vec
        clin[i] = scaler.transform_row(row)
        hfpef[i] = float(row["hfpef_%"])
        grade[i] = int(row["dd_grade_index"])

    return heart, liver, clin, hfpef, grade


def build_model(learning_rate: float = 5e-4) -> tf.keras.Model:
    heart_in = tf.keras.Input(shape=(64,), name="heart_input")
    liver_in = tf.keras.Input(shape=(128,), name="liver_input")
    clin_in = tf.keras.Input(shape=(9,), name="clinical_input")

    # Heart branch: 64 -> 32
    h = tf.keras.layers.Dense(32)(heart_in)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    # Liver branch: 128 -> 128 (Identity-like mapping or further compression)
    l = tf.keras.layers.Dense(128)(liver_in)
    l = tf.keras.layers.BatchNormalization()(l)
    l = tf.keras.layers.ReLU()(l)

    # Clinical branch: 9 -> 32
    c = tf.keras.layers.Dense(32)(clin_in)
    c = tf.keras.layers.BatchNormalization()(c)
    c = tf.keras.layers.ReLU()(c)

    # Fusion core: 192 -> 256 -> 128
    x = tf.keras.layers.Concatenate()([h, l, c])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    hfpef_out = tf.keras.layers.Dense(1, name="hfpef_head")(x)
    grade_logits = tf.keras.layers.Dense(4, name="grade_logits")(x)
    grade_out = tf.keras.layers.Activation("softmax", name="grade_head")(grade_logits)

    model = tf.keras.Model(
        inputs=[heart_in, liver_in, clin_in],
        outputs={"hfpef_head": hfpef_out, "grade_head": grade_out},
        name="ClinicalFusionModelKeras",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "hfpef_head": tf.keras.losses.Huber(),
            "grade_head": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={"hfpef_head": 0.5, "grade_head": 1.0},
        metrics={
            "hfpef_head": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "grade_head": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        },
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="fusion_training_dataset.csv")
    parser.add_argument("--heart_dir", type=str, default="heart_vectors")
    parser.add_argument("--liver_dir", type=str, default="liver_vectors")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_path", type=str, default="best_fusion_model.keras")
    args = parser.parse_args()

    set_seed(args.seed)
    print("[INFO] TensorFlow:", tf.__version__)
    print("[INFO] GPUs:", tf.config.list_physical_devices("GPU"))

    df = prepare_dataframe(args.csv_path)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=df["dd_grade_index"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    scaler = ClinicalScaler().fit(train_df)

    print(f"[INFO] Rows total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)}")
    print("[INFO] Train class distribution:")
    print(train_df["dd_grade_index"].value_counts().sort_index())
    print("[INFO] Val class distribution:")
    print(val_df["dd_grade_index"].value_counts().sort_index())

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2, 3]),
        y=train_df["dd_grade_index"].to_numpy(),
    ).astype(np.float32)
    class_weights_map = {i: float(w) for i, w in enumerate(class_weights_arr)}
    print("[INFO] Class weights:", class_weights_map)

    Xh_train, Xl_train, Xc_train, yreg_train, ycls_train = load_vectors_and_targets(
        train_df, args.heart_dir, args.liver_dir, scaler
    )
    Xh_val, Xl_val, Xc_val, yreg_val, ycls_val = load_vectors_and_targets(
        val_df, args.heart_dir, args.liver_dir, scaler
    )

    model = build_model(learning_rate=float(args.learning_rate))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.out_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger("train_fusion_keras.log.csv", append=False),
    ]

    # per-sample weights for classification output only
    sample_w_cls_train = np.array([class_weights_map[int(g)] for g in ycls_train], dtype=np.float32)
    sample_w_cls_val = np.array([class_weights_map[int(g)] for g in ycls_val], dtype=np.float32)

    history = model.fit(
        x=[Xh_train, Xl_train, Xc_train],
        y={
            "hfpef_head": yreg_train,
            "grade_head": ycls_train,
        },
        sample_weight={
            "hfpef_head": np.ones_like(yreg_train, dtype=np.float32),
            "grade_head": sample_w_cls_train,
        },
        validation_data=(
            [Xh_val, Xl_val, Xc_val],
            {
                "hfpef_head": yreg_val,
                "grade_head": ycls_val,
            },
            {
                "hfpef_head": np.ones_like(yreg_val, dtype=np.float32),
                "grade_head": sample_w_cls_val,
            },
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final history
    pd.DataFrame(history.history).to_csv("train_fusion_keras_history.csv", index=False)

    # Load best model if present
    if os.path.exists(args.out_path):
        model = tf.keras.models.load_model(args.out_path)
        print(f"[INFO] Loaded best model from: {args.out_path}")

    # Evaluation
    pred_outputs = model.predict([Xh_val, Xl_val, Xc_val], batch_size=256, verbose=0)
    pred_reg = pred_outputs["hfpef_head"].reshape(-1)
    pred_prob = pred_outputs["grade_head"]
    pred_cls = np.argmax(pred_prob, axis=1)

    val_acc = accuracy_score(ycls_val, pred_cls)
    val_mae = mean_absolute_error(yreg_val, pred_reg)

    print(f"[RESULT] Best-val classification accuracy: {val_acc * 100:.2f}%")
    print(f"[RESULT] Best-val regression MAE: {val_mae:.4f}")
    print("\n[RESULT] Classification report:")
    print(
        classification_report(
            ycls_val,
            pred_cls,
            target_names=["Normal", "Grade I", "Grade II", "Grade III"],
            digits=4,
        )
    )

    # Save lightweight eval artifacts
    pd.DataFrame({
        "true_grade": ycls_val,
        "pred_grade": pred_cls,
        "true_hfpef": yreg_val,
        "pred_hfpef": pred_reg,
    }).to_csv("keras_val_predictions.csv", index=False)

    np.savez(
        "keras_val_probabilities.npz",
        y_true_grade=ycls_val,
        y_pred_prob=pred_prob,
        y_true_hfpef=yreg_val,
        y_pred_hfpef=pred_reg,
    )

    print("[DONE] Saved:")
    print("  -", args.out_path)
    print("  - train_fusion_keras.log.csv")
    print("  - train_fusion_keras_history.csv")
    print("  - keras_val_predictions.csv")
    print("  - keras_val_probabilities.npz")


if __name__ == "__main__":
    main()