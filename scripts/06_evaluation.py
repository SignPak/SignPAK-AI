"""
06_evaluation.py — SignPAK-AI (726-Dim Standalone Offline Evaluator)
=====================================================================
Evaluates saved checkpoints from models/checkpoints_opt/ on test.csv.
Computes Classification Report (Precision, Recall, F1) and Confusion Matrix.

Run from: SIGNPAK-AI root → python scripts/06_evaluation.py
"""

import csv
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

_THIS        = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CSV_DIR      = DATA_DIR / "csv"
CKPT_DIR     = PROJECT_ROOT / "models" / "checkpoints_opt"

MAX_SEQ_LEN = 60
FEATURE_DIM = 726
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def compute_features_vectorized(raw_arr: np.ndarray) -> np.ndarray:
    T = raw_arr.shape[0]
    coords = raw_arr.reshape(T, 75, 3).copy()

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


class IndependentTestDataset(Dataset):
    def __init__(self, csv_path: Path, label_to_idx: dict, max_len: int = MAX_SEQ_LEN):
        self.max_len = max_len
        self.samples = []
        self.label_to_idx = label_to_idx

        if not csv_path.exists():
            raise FileNotFoundError(f"Test manifest CSV not found at: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lm_path = row.get("landmark_path", "")
                if lm_path and (PROJECT_ROOT / lm_path).exists():
                    label_str = row["label"]
                    if label_str in self.label_to_idx:
                        self.samples.append({
                            "file": PROJECT_ROOT / lm_path,
                            "label": self.label_to_idx[label_str]
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_data = np.load(str(item["file"]))

        t = raw_data.shape[0]
        if t != self.max_len:
            indices = np.linspace(0, t - 1, self.max_len).astype(int)
            raw_data = raw_data[indices]

        processed_data = compute_features_vectorized(raw_data)
        return torch.from_numpy(processed_data), torch.tensor(item["label"], dtype=torch.long)


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
    def __init__(self, feature_dim: int = FEATURE_DIM, num_classes: int = 28):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(MAX_SEQ_LEN),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.conv1 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(256)
        self.relu  = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)

        self.lstm = nn.LSTM(
            input_size=256, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.5
        )

        self.attn = TemporalAttention(256)
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
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


def evaluate_checkpoint(ckpt_name: str, label_to_idx: dict):
    ckpt_path = CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"⚠️ Checkpoint file '{ckpt_name}' not found at {ckpt_path}. Skipping...")
        return

    num_classes = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    print(f"\n{'='*65}")
    print(f" 📊 EVALUATING CHECKPOINT: {ckpt_name}")
    print(f"{'='*65}")

    test_path = CSV_DIR / "test.csv"
    test_dataset = IndependentTestDataset(test_path, label_to_idx=label_to_idx)

    if len(test_dataset) == 0:
        print("⚠️ No valid test samples found matching label_map.json.")
        return

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = SignPAKClassifierOpt(feature_dim=FEATURE_DIM, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    target_names = [idx_to_label[i] for i in range(num_classes)]

    print("\n📋 Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))

    cm = confusion_matrix(all_targets, all_preds)
    print("🧩 Confusion Matrix:")
    print(cm)
    print("=" * 65)


def main():
    print(f"🚀 Independent Evaluation Engine | Target Device: {DEVICE}")

    label_map_path = CKPT_DIR / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found at {label_map_path}. Run training script 05 first.")

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)

    # Evaluate best fold model from checkpoints_opt
    for ckpt in CKPT_DIR.glob("best_opt_model_fold*.pth"):
        evaluate_checkpoint(ckpt.name, label_to_idx)


if __name__ == "__main__":
    main()