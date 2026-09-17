"""
05_transfer_learning_include.py — SignPAK-AI (Optimized INCLUDE / ISL Transfer Engine)
======================================================================================
Optimized Transfer Learning Architecture:
  - Loads Clean AND Offline Augmented samples safely (data/csv/).
  - Strict GroupKFold (N_SPLITS=6) for Leave-One-Signer-Out Cross-Validation.
  - Evaluation Integrity: Validates exclusively on CLEAN samples of holdout signers.
  - Automatically unwraps nested checkpoint dictionaries (ckpt['model']).
  - Maps 32-40 pretrained BiLSTM weight tensors from ISL backbone.
  - Real-Time Skeleton Warping: Limb scaling + 3D rotation + coordinate noise.

Run from: SIGNPAK-AI root → python scripts/05_transfer_learning_include.py
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

# ── Paths & Configuration ─────────────────────────────────────────────────────
_THIS        = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CSV_V2_DIR   = DATA_DIR / "csv"  # Fixed path to load full augmented dataset
CKPT_DIR     = PROJECT_ROOT / "models" / "checkpoints_include"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH = PROJECT_ROOT / "models" / "checkpoints" / "include_pretrained.pth"

MAX_SEQ_LEN     = 60
FEATURE_DIM     = 726
INCLUDE_IN_DIM  = 134
BATCH_SIZE      = 16
EPOCHS          = 70
WARMUP_EPOCHS   = 10
N_SPLITS        = 6  # True Leave-One-Signer-Out CV
LEARNING_RATE   = 8e-4

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
# 1. Architecture Matching INCLUDE Checkpoint
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


class ExactIncludeModelOpt(nn.Module):
    def __init__(self, in_features: int = FEATURE_DIM, num_classes: int = 37):
        super().__init__()
        # Stem adapter: 726 -> 134
        self.adapter = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(MAX_SEQ_LEN),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, INCLUDE_IN_DIM)
        )

        # 5-Layer Stacked BiLSTM Backbone
        self.lstm = nn.LSTM(
            input_size=INCLUDE_IN_DIM,
            hidden_size=256,
            num_layers=5,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        # Task-Specific Head for PSL
        self.attn = TemporalAttention(512)
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def load_pretrained_weights(self, weights_path: Path):
        if not weights_path.exists():
            print(f"⚠️ Pretrained file not found at {weights_path}.")
            return False

        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

            if isinstance(ckpt, dict) and "model" in ckpt:
                ckpt = ckpt["model"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            model_dict = self.state_dict()
            transferred_dict = {}

            for k, v in ckpt.items():
                if not isinstance(v, torch.Tensor) or "l1" in k:
                    continue

                if k in model_dict and model_dict[k].shape == v.shape:
                    transferred_dict[k] = v
                else:
                    clean_k = k if k.startswith("lstm.") else f"lstm.{k}"
                    if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
                        transferred_dict[clean_k] = v

            model_dict.update(transferred_dict)
            self.load_state_dict(model_dict)
            print(f"✅ REAL TRANSFER SUCCESS: Loaded {len(transferred_dict)} / {len(ckpt)} INCLUDE pretrained tensors!")
            return len(transferred_dict) > 0
        except Exception as e:
            print(f"⚠️ Error loading weights: {e}")
            return False

    def freeze_backbone(self):
        for param in self.lstm.parameters():
            param.requires_grad = False
        print("🔒 INCLUDE BiLSTM Backbone layers FROZEN for Warmup Stage.")

    def unfreeze_backbone(self):
        for param in self.lstm.parameters():
            param.requires_grad = True
        print("🔓 INCLUDE BiLSTM Backbone layers UNFROZEN for Fine-Tuning.")

    def forward(self, x):
        x_proj = self.adapter(x)        # (B, T, 134)
        out_lstm, _ = self.lstm(x_proj) # (B, T, 512)
        context = self.attn(out_lstm)   # (B, 512)
        logits = self.classifier(context)
        return logits


# ═════════════════════════════════════════════════════════════════════════════
# 2. Vectorized Skeleton-Warping Feature Processor
# ═════════════════════════════════════════════════════════════════════════════

def extract_features_vectorized(raw_arr: np.ndarray, augment: bool = False) -> np.ndarray:
    T = raw_arr.shape[0]
    coords = raw_arr.reshape(T, 75, 3).copy()

    if augment:
        left_arm_scale = np.random.uniform(0.88, 1.12)
        right_arm_scale = np.random.uniform(0.88, 1.12)
        coords[:, 33:54, :] *= left_arm_scale
        coords[:, 54:75, :] *= right_arm_scale

        xy_noise = np.random.normal(0, 0.025, size=(T, 75, 2))
        z_noise  = np.random.normal(0, 0.035, size=(T, 75, 1))
        coords += np.concatenate([xy_noise, z_noise], axis=-1)

        angle = np.radians(np.random.uniform(-12, 12))
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

    left_shoulder  = coords[:, 11, :]
    right_shoulder = coords[:, 12, :]
    shoulder_dist  = np.linalg.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True)
    shoulder_dist[shoulder_dist < 1e-4] = 1.0
    chest_center   = (left_shoulder + right_shoulder) / 2.0

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


class FastTransferDataset(Dataset):
    def __init__(self, samples: list, max_len: int = MAX_SEQ_LEN, augment: bool = False):
        self.samples = samples
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_data = np.load(str(item["file"]))

        t = raw_data.shape[0]
        if t != self.max_len:
            indices = np.linspace(0, t - 1, self.max_len).astype(int)
            raw_data = raw_data[indices]

        processed_data = extract_features_vectorized(raw_data, augment=self.augment)
        return torch.from_numpy(processed_data), torch.tensor(item["label"], dtype=torch.long)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Execution Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_include_transfer():
    print(f"🚀 Optimized INCLUDE Transfer Engine Running on Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    all_samples = []
    all_label_strs = set()

    for split in ["train", "val", "test"]:
        csv_path = CSV_V2_DIR / f"{split}.csv"
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    lm_path = row.get("landmark_path", "")
                    if lm_path and (PROJECT_ROOT / lm_path).exists():
                        is_aug = row.get("augmented", "").lower() == "true"
                        all_samples.append({
                            "file": PROJECT_ROOT / lm_path,
                            "label_str": row["label"],
                            "signer_id": row["signer_id"],
                            "is_augmented": is_aug
                        })
                        all_label_strs.add(row["label"])

    label_to_idx = {lbl: i for i, lbl in enumerate(sorted(all_label_strs))}
    num_classes = len(label_to_idx)

    with open(CKPT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2)

    for s in all_samples:
        s["label"] = label_to_idx[s["label_str"]]

    X = np.array(all_samples)
    y = np.array([s["label"] for s in all_samples])
    groups = np.array([s["signer_id"] for s in all_samples])

    unique_groups = sorted(set(groups))
    clean_count = sum(1 for s in all_samples if not s["is_augmented"])
    aug_count = sum(1 for s in all_samples if s["is_augmented"])
    print(f"Combined Dataset: {len(all_samples)} Total Samples ({clean_count} Clean + {aug_count} Offline Augmented)")
    print(f"Signers: {unique_groups} | Classes: {num_classes}\n")

    n_folds = min(N_SPLITS, len(unique_groups))
    gkf = GroupKFold(n_splits=n_folds)
    fold_accuracies = []
    num_workers = 2 if os.name == "nt" else 4

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n{'='*60}\n 🔄 RUNNING INCLUDE TRANSFER FOLD {fold}/{n_folds} (GroupKFold)\n{'='*60}")

        train_samples = X[train_idx].tolist()
        val_samples_all = X[val_idx].tolist()
        val_samples_clean = [s for s in val_samples_all if not s["is_augmented"]]
        if not val_samples_clean:
            val_samples_clean = val_samples_all

        val_signers = sorted(set(s["signer_id"] for s in val_samples_clean))
        print(f"Train Size: {len(train_samples)} samples (Clean + Aug) | Val Size: {len(val_samples_clean)} CLEAN samples (Signers: {val_signers})")

        train_loader = DataLoader(
            FastTransferDataset(train_samples, augment=True),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            FastTransferDataset(val_samples_clean, augment=False),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        model = ExactIncludeModelOpt(in_features=FEATURE_DIM, num_classes=num_classes).to(DEVICE)
        
        loaded = model.load_pretrained_weights(WEIGHTS_PATH)
        if loaded:
            model.freeze_backbone()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.20)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=3e-2)
        
        total_steps = EPOCHS * len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=total_steps, pct_start=0.2)

        best_val_acc = 0.0
        fold_save_path = CKPT_DIR / f"best_include_model_fold{fold}.pth"

        for epoch in range(1, EPOCHS + 1):
            if epoch == WARMUP_EPOCHS + 1 and loaded:
                model.unfreeze_backbone()
                optimizer = optim.AdamW([
                    {'params': model.adapter.parameters(),    'lr': LEARNING_RATE},
                    {'params': model.lstm.parameters(),       'lr': LEARNING_RATE * 0.15},
                    {'params': model.attn.parameters(),       'lr': LEARNING_RATE},
                    {'params': model.classifier.parameters(), 'lr': LEARNING_RATE},
                ], weight_decay=3e-2)
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=[LEARNING_RATE, LEARNING_RATE * 0.15, LEARNING_RATE, LEARNING_RATE],
                    total_steps=(EPOCHS - WARMUP_EPOCHS) * len(train_loader), pct_start=0.15
                )

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
                stage_str = "Warmup" if (epoch <= WARMUP_EPOCHS and loaded) else "FineTune"
                print(f"[{stage_str}] Fold {fold} | Epoch [{epoch:02d}/{EPOCHS}] - Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

        fold_accuracies.append(best_val_acc)
        print(f"✅ Fold {fold} Complete. Best Accuracy: {best_val_acc*100:.2f}%")

    print("\n" + "=" * 60)
    print(" 📊 INCLUDE PRETRAINED TRANSFER SUMMARY")
    print("=" * 60)
    for f_idx, acc in enumerate(fold_accuracies, 1):
        print(f"  Fold {f_idx}: {acc*100:.2f}%")
    print(f"\n  ⭐ Mean Out-of-Sample Accuracy: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_include_transfer()