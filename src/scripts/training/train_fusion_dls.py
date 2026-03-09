import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


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


@dataclass
class ClinicalScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "ClinicalScaler":
        values = df[CLINICAL_COLUMNS].to_numpy(dtype=np.float32)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def transform_row(self, row: pd.Series) -> np.ndarray:
        x = np.array([row[col] for col in CLINICAL_COLUMNS], dtype=np.float32)
        return (x - self.mean) / self.std


class CDSSFusionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, heart_dir: str, liver_dir: str, scaler: ClinicalScaler):
        self.df = dataframe.reset_index(drop=True)
        self.heart_dir = heart_dir
        self.liver_dir = liver_dir
        self.scaler = scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        heart_path = os.path.join(self.heart_dir, f"{row['heart_feature_key']}.npy")
        liver_path = os.path.join(self.liver_dir, f"{row['liver_feature_key']}.npy")

        if not os.path.exists(heart_path):
            raise FileNotFoundError(f"Missing heart vector: {heart_path}")
        if not os.path.exists(liver_path):
            raise FileNotFoundError(f"Missing liver vector: {liver_path}")

        h_vec = np.load(heart_path).astype(np.float32).reshape(-1)
        l_vec = np.load(liver_path).astype(np.float32).reshape(-1)

        if h_vec.shape[0] != 64:
            raise ValueError(f"Heart vector must be 64D, got {h_vec.shape[0]} at {heart_path}")
        if l_vec.shape[0] != 768:
            raise ValueError(f"Liver vector must be 768D, got {l_vec.shape[0]} at {liver_path}")

        clinical_vec = self.scaler.transform_row(row)

        return (
            torch.from_numpy(h_vec),
            torch.from_numpy(l_vec),
            torch.from_numpy(clinical_vec),
            torch.tensor([float(row["hfpef_%"])], dtype=torch.float32),
            torch.tensor(int(row["dd_grade_index"]), dtype=torch.long),
        )


class ClinicalFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.heart_branch = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.liver_branch = nn.Sequential(
            nn.Linear(768, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.clin_branch = nn.Sequential(
            nn.Linear(9, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.fusion_core = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.hfpef_head = nn.Linear(128, 1)
        self.grade_head = nn.Linear(128, 4)

    def forward(self, h, l, c):
        h_f = self.heart_branch(h)
        l_f = self.liver_branch(l)
        c_f = self.clin_branch(c)

        combined = torch.cat((h_f, l_f, c_f), dim=1)
        reasoning = self.fusion_core(combined)

        return self.hfpef_head(reasoning), self.grade_head(reasoning)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    if "liver_stage_idx" not in df.columns:
        if "liver_stage" not in df.columns:
            raise ValueError("CSV must include either 'liver_stage_idx' or 'liver_stage'.")
        df["liver_stage_idx"] = df["liver_stage"].astype(str).str.extract(r"(\d)").astype(int)

    required = [
        "heart_feature_key",
        "liver_feature_key",
        *CLINICAL_COLUMNS,
        "hfpef_%",
        "dd_grade_index",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    if df[required].isna().any().any():
        bad = df[required].isna().sum()
        offenders = bad[bad > 0].to_dict()
        raise ValueError(f"Null values found in required columns: {offenders}")

    df["dd_grade_index"] = df["dd_grade_index"].astype(int)
    if not set(df["dd_grade_index"].unique()).issubset({0, 1, 2, 3}):
        raise ValueError("dd_grade_index must only contain 0,1,2,3")

    return df


def build_loaders(
    df: pd.DataFrame,
    heart_dir: str,
    liver_dir: str,
    batch_size: int,
    seed: int,
):
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["dd_grade_index"],
    )

    scaler = ClinicalScaler.fit(train_df)

    train_ds = CDSSFusionDataset(train_df, heart_dir, liver_dir, scaler)
    val_ds = CDSSFusionDataset(val_df, heart_dir, liver_dir, scaler)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, train_df, val_df


def train(
    csv_path: str = "fusion_training_dataset.csv",
    heart_dir: str = "heart_vectors",
    liver_dir: str = "liver_vectors",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    out_path: str = "best_fusion_model.pth",
    patience: int = 7,
    seed: int = 42,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    df = prepare_dataframe(csv_path)
    train_loader, val_loader, train_df, val_df = build_loaders(
        df=df,
        heart_dir=heart_dir,
        liver_dir=liver_dir,
        batch_size=batch_size,
        seed=seed,
    )

    print(f"Rows total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)}")
    print("Train class distribution:")
    print(train_df["dd_grade_index"].value_counts().sort_index())
    print("Val class distribution:")
    print(val_df["dd_grade_index"].value_counts().sort_index())

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2, 3]),
        y=train_df["dd_grade_index"].to_numpy(),
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"Class weights: {class_weights}")

    model = ClinicalFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion_reg = nn.HuberLoss()
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights_t)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for h, l, c, target_p, target_g in train_loader:
            h = h.to(device)
            l = l.to(device)
            c = c.to(device)
            target_p = target_p.to(device)
            target_g = target_g.to(device)

            optimizer.zero_grad()
            pred_p, pred_g = model(h, l, c)

            loss_reg = criterion_reg(pred_p, target_p)
            loss_cls = criterion_cls(pred_g, target_g)
            loss = loss_cls + (0.5 * loss_reg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / max(len(train_loader), 1)

        model.eval()
        val_loss_sum = 0.0
        val_targets_cls = []
        val_preds_cls = []
        val_targets_reg = []
        val_preds_reg = []

        with torch.no_grad():
            for h, l, c, target_p, target_g in val_loader:
                h = h.to(device)
                l = l.to(device)
                c = c.to(device)
                target_p = target_p.to(device)
                target_g = target_g.to(device)

                pred_p, pred_g = model(h, l, c)
                loss_reg = criterion_reg(pred_p, target_p)
                loss_cls = criterion_cls(pred_g, target_g)
                loss = loss_cls + (0.5 * loss_reg)
                val_loss_sum += loss.item()

                pred_class = torch.argmax(pred_g, dim=1)
                val_preds_cls.extend(pred_class.cpu().numpy().tolist())
                val_targets_cls.extend(target_g.cpu().numpy().tolist())
                val_preds_reg.extend(pred_p.squeeze(1).cpu().numpy().tolist())
                val_targets_reg.extend(target_p.squeeze(1).cpu().numpy().tolist())

        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        val_acc = accuracy_score(val_targets_cls, val_preds_cls)
        val_mae = mean_absolute_error(val_targets_reg, val_preds_reg)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "class_weights": class_weights,
                    "seed": seed,
                },
                out_path,
            )
            print(f"[INFO] Checkpoint improved. Saved to: {out_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs")

        if epochs_no_improve >= patience:
            print("[INFO] Early stopping triggered")
            break

    print("\n[INFO] Loading best checkpoint for final validation report...")
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    final_targets = []
    final_preds = []
    with torch.no_grad():
        for h, l, c, _, target_g in val_loader:
            h = h.to(device)
            l = l.to(device)
            c = c.to(device)

            _, pred_g = model(h, l, c)
            pred_class = torch.argmax(pred_g, dim=1)
            final_preds.extend(pred_class.cpu().numpy().tolist())
            final_targets.extend(target_g.numpy().tolist())

    print("\n[INFO] Final Classification Report (Validation Set):")
    print(
        classification_report(
            final_targets,
            final_preds,
            labels=[0, 1, 2, 3],
            target_names=["Normal", "Grade I", "Grade II", "Grade III"],
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fusion DLS model with production-grade validation safeguards")
    parser.add_argument("--csv_path", type=str, default="fusion_training_dataset.csv")
    parser.add_argument("--heart_dir", type=str, default="heart_vectors")
    parser.add_argument("--liver_dir", type=str, default="liver_vectors")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--out_path", type=str, default="best_fusion_model.pth")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(
        csv_path=args.csv_path,
        heart_dir=args.heart_dir,
        liver_dir=args.liver_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        out_path=args.out_path,
        patience=args.patience,
        seed=args.seed,
    )
