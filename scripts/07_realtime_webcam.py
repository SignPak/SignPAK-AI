"""
07_realtime_webcam.py — SignPAK-AI (Live 726-Dim Kinematic Webcam Engine)
==========================================================================
Captures webcam frames in real time, extracts MediaPipe landmarks,
transforms them into the full 726-dimensional kinematic vector, and runs
inference using trained checkpoints from models/checkpoints_opt/.

Run from: SIGNPAK-AI root → python scripts/07_realtime_webcam.py
"""

import cv2
import json
import torch
import torch.nn as nn
import collections
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Paths & Config ────────────────────────────────────────────────────────────
_THIS        = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
MODEL_DIR    = PROJECT_ROOT / "models"
CKPT_DIR     = MODEL_DIR / "checkpoints_opt"

POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
LABEL_MAP_PATH  = CKPT_DIR / "label_map.json"

FEATURE_DIM = 726
MAX_SEQ_LEN = 60
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


# ═════════════════════════════════════════════════════════════════════════════
# 726-Dimensional Kinematic Feature Transformation Engine
# ═════════════════════════════════════════════════════════════════════════════

def compute_features_vectorized(raw_arr: np.ndarray) -> np.ndarray:
    """Transforms raw 225-dim sequence array (T, 225) into full 726-dim kinematic vector."""
    T = raw_arr.shape[0]
    coords = raw_arr.reshape(T, 75, 3).copy()

    # Calculate Joint Bending Angles
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

    # Calculate Bone Distance Matrices
    dist_list = []
    for p1, p2 in DIST_PAIRS:
        d = np.linalg.norm(coords[:, p1, :] - coords[:, p2, :], axis=-1, keepdims=True)
        dist_list.append(d)
    
    distances = np.concatenate(dist_list, axis=-1)
    if distances.shape[1] < 30:
        distances = np.pad(distances, ((0, 0), (0, 30 - distances.shape[1])))

    # Torso Centering and Normalization
    left_shoulder  = coords[:, 11, :]
    right_shoulder = coords[:, 12, :]
    shoulder_dist  = np.linalg.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True)
    shoulder_dist[shoulder_dist < 1e-4] = 1.0
    chest_center   = (left_shoulder + right_shoulder) / 2.0

    coords[:, :33, :] = (coords[:, :33, :] - chest_center[:, None, :]) / shoulder_dist[:, None, :]

    # Left Hand Wrist-Relative Normalization
    lh = coords[:, 33:54, :]
    lh_wrist = lh[:, 0:1, :].copy()
    lh_scale = np.linalg.norm(lh[:, 0, :] - lh[:, 9, :], axis=-1, keepdims=True)
    lh_scale[lh_scale < 1e-4] = 1.0
    coords[:, 33:54, :] = (lh - lh_wrist) / lh_scale[:, None, :]

    # Right Hand Wrist-Relative Normalization
    rh = coords[:, 54:75, :]
    rh_wrist = rh[:, 0:1, :].copy()
    rh_scale = np.linalg.norm(rh[:, 0, :] - rh[:, 9, :], axis=-1, keepdims=True)
    rh_scale[rh_scale < 1e-4] = 1.0
    coords[:, 54:75, :] = (rh - rh_wrist) / rh_scale[:, None, :]

    # Compute Velocity and Acceleration
    normalized_pos = coords.reshape(T, 225)
    velocity = np.zeros_like(normalized_pos)
    velocity[1:] = normalized_pos[1:] - normalized_pos[:-1]

    accel = np.zeros_like(velocity)
    accel[1:] = velocity[1:] - velocity[:-1]

    return np.concatenate([normalized_pos, velocity, accel, angles, distances], axis=-1).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Classifier Model Architecture (Matches Script 05)
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


# ═════════════════════════════════════════════════════════════════════════════
# Live Webcam Engine
# ═════════════════════════════════════════════════════════════════════════════

def run_live_webcam():
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Label map not found at {LABEL_MAP_PATH}. Run script 05 first.")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes  = len(label_to_idx)

    # Automatically select best available fold checkpoint from models/checkpoints_opt/
    ckpt_path = None
    for candidate in sorted(CKPT_DIR.glob("best_opt_model_fold*.pth")):
        ckpt_path = candidate
        break

    if not ckpt_path or not ckpt_path.exists():
        raise FileNotFoundError(f"No trained checkpoints found in {CKPT_DIR}. Run script 05 first.")

    print(f"📦 Loading Trained Model Weights: {ckpt_path.name}")
    print(f"📊 Vocabulary Size: {num_classes} classes")
    
    model = SignPAKClassifierOpt(feature_dim=FEATURE_DIM, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()

    # Initialize MediaPipe Task Engines
    base_pose_opts = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    pose_opts = vision.PoseLandmarkerOptions(base_options=base_pose_opts, running_mode=vision.RunningMode.IMAGE)
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

    base_hand_opts = python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    hand_opts = vision.HandLandmarkerOptions(base_options=base_hand_opts, running_mode=vision.RunningMode.IMAGE, num_hands=2)
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Web camera could not be opened.")
        return

    print("\n🎥 Real-Time Sign Language Recognition Active!")
    print("   Press 'q' or 'ESC' on the camera window to quit.\n")

    frame_buffer = collections.deque(maxlen=MAX_SEQ_LEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror view
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        row_vec = np.zeros(225, dtype=np.float32)

        # Pose Extraction (33 landmarks x 3 = 99 floats)
        pose_res = pose_landmarker.detect(mp_image)
        if pose_res.pose_landmarks and len(pose_res.pose_landmarks[0]) >= 33:
            for idx, lm in enumerate(pose_res.pose_landmarks[0][:33]):
                row_vec[idx * 3 : idx * 3 + 3] = [lm.x, lm.y, lm.z]

        # Hand Extraction (21 landmarks x 3 x 2 hands = 126 floats)
        hand_res = hand_landmarker.detect(mp_image)
        if hand_res.hand_landmarks and hand_res.handedness:
            for hand_lms, handedness_cat in zip(hand_res.hand_landmarks, hand_res.handedness):
                lbl = handedness_cat[0].category_name.lower()
                offset = 99 if lbl == "left" else 162
                for idx, lm in enumerate(hand_lms[:21]):
                    row_vec[offset + idx * 3 : offset + idx * 3 + 3] = [lm.x, lm.y, lm.z]

        frame_buffer.append(row_vec)

        predicted_sign = "Buffering..."
        confidence = 0.0

        if len(frame_buffer) == MAX_SEQ_LEN:
            raw_buffer = np.array(frame_buffer)  # (60, 225)
            feat_buffer = compute_features_vectorized(raw_buffer)  # (60, 726)
            tensor_in = torch.tensor(feat_buffer, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(tensor_in)
                probs  = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

                confidence = conf.item()
                if confidence > 0.40:
                    predicted_sign = idx_to_label[pred_idx.item()]
                else:
                    predicted_sign = "Listening..."

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        display_str = f"SIGN: {predicted_sign.upper()} ({confidence*100:.1f}%)"
        cv2.putText(frame, display_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("SignPAK-AI Live Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()
    hand_landmarker.close()


if __name__ == "__main__":
    run_live_webcam()