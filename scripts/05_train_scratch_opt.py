"""
05_train_scratch_opt.py — SignPAK-AI (Dynamic LOSO Engine + Confusion Matrix)
=============================================================================
Features:
  - Dynamic LOSO Cross-Validation: Automatically scales across all N signers.
  - 726-dim Kinematic Feature Vector (Normalized Coordinates + Velocity + Accel + Angles + Distances).
  - Smart Augmentation Routing: Prevents excessive double-distortion.
  - Generates Full Confusion Matrix & Top Misclassified Word Pairs per fold.

Run from: SIGNPAK-AI root → python scripts/05_train_scratch_opt.py
"""

import os
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

# ── Paths & Hyperparameters ──────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_DIR = DATA_DIR / "csv"
CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints_opt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 60
FEATURE_DIM = 726
BATCH_SIZE = 16
EPOCHS = 70
LEARNING_RATE = 8e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANGLE_TRIPLETS = [
    (11, 13, 15), (12, 14, 16), (13, 15, 33), (14, 16, 54),
    (33, 34, 37), (33, 38, 41), (33, 42, 45), (54, 55, 58),
    (54, 59, 62), (54, 63, 66),
]

DIST_PAIRS = [
    (15, 16), (11, 13), (12, 14), (13, 15), (14, 16),
    (33, 37), (33, 41), (33, 45), (33, 49), (54, 58),
    (54, 62), (54, 66), (54, 70), (37, 41), (41, 45),
    (45, 49), (58, 62), (62, 66), (66, 70),
]


# ═════════════════════════════════════════════════════════════════════════════
# 726-dim Vectorized Kinematic Feature Extraction
# ═════════════════════════════════════════════════════════════════════════════

def compute_features_vectorized(raw_arr: np.ndarray, augment: bool = False) -> np.ndarray:
    T = raw_arr.shape[0]
    coords = raw_arr.reshape(T, 75, 3).copy()

    if augment:
        left_arm_scale = np.random.uniform(0.92, 1.08)
        right_arm_scale = np.random.uniform(0.92, 1.08)
        coords[:, 33:54, :] *= left_arm_scale
        coords[:, 54:75, :] *= right_arm_scale

        xy_noise = np.random.normal(0, 0.015, size=(T, 75, 2))
        z_noise = np.random.normal(0, 0.020, size=(T, 75, 1))
        coords += np.concatenate([xy_noise, z_noise], axis=-1)

        angle = np.radians(np.random.uniform(-8, 8))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_m = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
        coords = np.matmul(coords, rot_m)

    angles_list = []
    for p1, p2, p3 in ANGLE_TRIPLETS:
        v1 = coords[:, p1, :] - coords[:, p2, :]
        v2 = coords[:, p3, :] - coords[:, p2, :]
        norm1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-6
        norm2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-6
        cos_a = np.sum(v1 * v2, axis=-1, keepdims=True) / (norm1 * norm2)
        ang = np.arccos(np.clip(cos_a, -1.0, 1.0))
        angles_list.append(ang)

    angles = np.concatenate(angles_list, axis=-1)
    if angles.shape[1] < 21:
        angles = np.pad(angles, ((0, 0), (0, 21 - angles.shape[1])))

    dist_list = []
    for p1, p2 in DIST_PAIRS:
        d = np.linalg.norm(coords[:, p1, :] - coords[:, p2, :], axis=-1, keepdims=True)
        dist_list.append(d)

    distances = np.concatenate(dist_list, axis=-1)
    if distances.shape[1] < 30:
        distances = np.pad(distances, ((0, 0), (0, 30 - distances.shape[1])))

    left_shoulder = coords[:, 11, :]
    right_shoulder = coords[:, 12, :]
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True)
    shoulder_dist[shoulder_dist < 1e-4] = 1.0
    chest_center = (left_shoulder + right_shoulder) / 2.0

    coords[:, :33, :] = (coords[:, :33, :] - chest_center[:, None, :]) / shoulder_dist[:, None, :]

    lh = coords[:, 33:54, :]
    lh_wrist = lh[:, 0:1, :].copy()
    lh_scale = np.linalg.norm(lh[:, 0, :] - lh[:, 9, :], axis=-1, keepdims=True)
    lh_scale[lh_scale < 1e-4] = 1.0
    coords[:, 33:54, :] = (lh - lh_wrist) / lh_scale[:, None, :]

    rh = coords[:, 54:75, :]
    rh_wrist = rh[:, 0:1, :].copy()
    rh_scale = np.linalg.norm(rh[:, 0, :] - rh[:, 9, :], axis=-1, keepdims=True)
    rh_scale[rh_scale < 1e-4] = 1.0
    coords[:, 54:75, :] = (rh - rh_wrist) / rh_scale[:, None, :]

    normalized_pos = coords.reshape(T, 225)
    velocity = np.zeros_like(normalized_pos)
    velocity[1:] = normalized_pos[1:] - normalized_pos[:-1]

    accel = np.zeros_like(velocity)
    accel[1:] = velocity[1:] - velocity[:-1]

    return np.concatenate([normalized_pos, velocity, accel, angles, distances], axis=-1).astype(np.float32)


class FastSignDataset(Dataset):
    def __init__(self, samples: list, max_len: int = MAX_SEQ_LEN, is_train: bool = False):
        self.samples = samples
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_data = np.load(str(item["file"]))

        t = raw_data.shape[0]
        if t != self.max_len:
            indices = np.linspace(0, t - 1, self.max_len).astype(int)
            raw_data = raw_data[indices]

        # Apply online warping ONLY if training AND sample is unaugmented original
        apply_online_warp = self.is_train and (not item.get("is_augmented", False))
        processed_data = compute_features_vectorized(raw_data, augment=apply_online_warp)

        return torch.from_numpy(processed_data), torch.tensor(item["label"], dtype=torch.long)


# ═════════════════════════════════════════════════════════════════════════════
# Model Architecture
# ═════════════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)


class SignPAKClassifierOpt(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, num_classes: int = 111):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(MAX_SEQ_LEN),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.conv1 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(0.4)

        self.lstm = nn.LSTM(
            input_size=256, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.4
        )

        self.attn = TemporalAttention(256)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = x.transpose(1, 2)
        x = self.drop1(self.relu(self.bn1(self.conv1(x))))
        x = x.transpose(1, 2)

        out, _ = self.lstm(x)
        context = self.attn(out)
        logits = self.classifier(context)
        return logits


# ═════════════════════════════════════════════════════════════════════════════
# Diagnostic Helper: Confusion Matrix
# ═════════════════════════════════════════════════════════════════════════════

def print_confusion_matrix_diagnostics(fold: int, y_true: list, y_pred: list, idx_to_label: dict, num_classes: int):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    print("\n" + "─" * 70)
    print(f" 🧩 CONFUSION MATRIX DIAGNOSTICS FOR FOLD {fold}")
    print("─" * 70)

    errors = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                actual_label = idx_to_label[i]
                pred_label = idx_to_label[j]
                count = cm[i, j]
                errors.append((actual_label, pred_label, count))

    if not errors:
        print("  🎉 PERFECT FOLD! Zero misclassifications recorded.")
    else:
        print(f"  ⚠️ Misclassifications Identified ({len(errors)} pairs):")
        errors = sorted(errors, key=lambda x: x[2], reverse=True)
        for actual_lbl, pred_lbl, cnt in errors[:15]:
            print(f"    • ACTUAL: [{actual_lbl:<20}] ──> PREDICTED: [{pred_lbl:<20}] ({cnt} time(s))")

    print("─" * 70 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# Execution Engine (LOSO Cross-Validation)
# ═════════════════════════════════════════════════════════════════════════════

def run_training():
    print(f"🚀 Scratch Training Engine running on: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    all_samples = []
    all_label_strs = set()

    # Load master manifest
    manifest_csv = CSV_DIR / "manifest.csv"
    csv_sources = [manifest_csv] if manifest_csv.exists() else [CSV_DIR / f"{s}.csv" for s in ["train", "val", "test"]]

    for csv_path in csv_sources:
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    signer_id = row.get("signer_id", "").strip()
                    lm_path = row.get("landmark_path", "").strip()

                    full_lm_path = PROJECT_ROOT / lm_path
                    if lm_path and full_lm_path.exists():
                        is_aug = row.get("augmented", "").lower() == "true"
                        all_samples.append({
                            "file": full_lm_path,
                            "label_str": row["label"],
                            "signer_id": signer_id,
                            "group_id": signer_id,
                            "is_augmented": is_aug
                        })
                        all_label_strs.add(row["label"])

    if not all_samples:
        print("❌ No landmark samples loaded. Ensure 04_augment_extract.py has run.")
        return

    label_to_idx = {lbl: i for i, lbl in enumerate(sorted(all_label_strs))}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)

    with open(CKPT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2)

    for s in all_samples:
        s["label"] = label_to_idx[s["label_str"]]

    X = np.array(all_samples)
    y = np.array([s["label"] for s in all_samples])
    groups = np.array([s["group_id"] for s in all_samples])

    unique_groups = sorted(set(groups))
    clean_count = sum(1 for s in all_samples if not s["is_augmented"])
    aug_count = sum(1 for s in all_samples if s["is_augmented"])

    print(f"Combined Dataset: {len(all_samples)} Total Samples ({clean_count} Clean + {aug_count} Offline Augmented)")
    print(f"👥 Active Signers ({len(unique_groups)} LOSO Folds): {unique_groups} | Target Classes: {num_classes}\n")

    n_folds = len(unique_groups)
    if n_folds < 2:
        print("⚠️ Less than 2 signers available. LOSO Cross-Validation requires at least 2 distinct signers.")
        return

    gkf = GroupKFold(n_splits=n_folds)
    fold_accuracies = []
    num_workers = 2 if os.name == "nt" else 4

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        val_group_name = unique_groups[fold - 1]
        print(f"\n{'='*60}\n 🔄 LOSO FOLD {fold}/{n_folds} ── Held-Out Validation Signer: [{val_group_name}]\n{'='*60}")

        train_samples = X[train_idx].tolist()
        
        # Validation strictly evaluates on CLEAN, unaugmented recordings of held-out signer
        val_samples_clean = [s for s in X[val_idx].tolist() if not s["is_augmented"]]
        if not val_samples_clean:
            val_samples_clean = X[val_idx].tolist()

        print(f"Train Size: {len(train_samples)} samples (Clean + Aug) | Val Size: {len(val_samples_clean)} Clean Samples")

        train_loader = DataLoader(
            FastSignDataset(train_samples, is_train=True),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            FastSignDataset(val_samples_clean, is_train=False),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        model = SignPAKClassifierOpt(feature_dim=FEATURE_DIM, num_classes=num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

        total_steps = EPOCHS * max(1, len(train_loader))
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=total_steps, pct_start=0.2)

        best_val_acc = 0.0
        fold_save_path = CKPT_DIR / f"best_opt_model_fold{fold}.pth"

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss, train_correct, total_train = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                total_train += labels.size(0)

            train_acc = train_correct / total_train if total_train > 0 else 0.0

            model.eval()
            val_loss, val_correct, total_val = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    total_val += labels.size(0)

            val_acc = val_correct / total_val if total_val > 0 else 0.0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), fold_save_path)

            if epoch % 10 == 0 or epoch == EPOCHS:
                print(f"Fold {fold} | Epoch [{epoch:02d}/{EPOCHS}] - Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

        # Load best fold checkpoint for confusion matrix diagnostics
        model.load_state_dict(torch.load(fold_save_path, map_location=DEVICE, weights_only=True))
        model.eval()
        fold_preds, fold_targets = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                fold_preds.extend(preds)
                fold_targets.extend(labels.numpy())

        print_confusion_matrix_diagnostics(fold, fold_targets, fold_preds, idx_to_label, num_classes)
        fold_accuracies.append(best_val_acc)
        print(f"✅ Fold {fold} Complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")

    print("\n" + "=" * 60)
    print(" 📊 LOSO CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for f_idx, (acc, grp) in enumerate(zip(fold_accuracies, unique_groups), 1):
        print(f"  Fold {f_idx} (Held-out: {grp}): {acc*100:.2f}%")
    print(f"\n  ⭐ Mean Out-of-Sample Accuracy: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_training()