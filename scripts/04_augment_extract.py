"""
04_augment_extract.py — SignPAK-AI (Max-Throughput Pixel-Domain Extraction)
===========================================================================
- Pinned 1-thread execution per worker to prevent CPU thrashing.
- Saturated 7 physical core workers with persistent MediaPipe instances.
- Precomputes 30x offline pixel augmentations with true optical landmark tracking.

Input  : data/cropped/<Signer_X>/<Category>/<label>.mp4
Output : data/landmarks/<Signer_X>/<label>/original.npy & aug_*.npy
"""

import os
import sys

# ── 1. Thread Pinning: Force 1 Thread Per Process (Prevents Lock Contention) ──
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["MEDIAPIPE_DISABLE_CLEARCUT"] = "1"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

def silence_cpp_stderr():
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 2)
        os.close(null_fd)
    except Exception:
        pass

silence_cpp_stderr()

import csv
import gc
import json
import random
import urllib.request
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CROPPED_DIR = DATA_DIR / "cropped"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
LANDMARK_DIR = DATA_DIR / "landmarks"
CSV_DIR = DATA_DIR / "csv"
LOG_DIR = DATA_DIR / "logs"
MODEL_DIR = PROJECT_ROOT / "models"

LOG_DIR.mkdir(parents=True, exist_ok=True)
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

POSE_LM = 33
HAND_LM = 21
COORD = 3
LM_DIM = POSE_LM * COORD + HAND_LM * COORD * 2  # 225

N_AUGMENTS = 30
RANDOM_SEED = 42

CSV_FIELDNAMES = [
    "signer_id", "split", "label", "label_idx",
    "category", "is_extra", "file_path",
    "landmark_path", "augmented", "aug_id",
]

# Persistent Global Worker Instances
_worker_pose_landmarker = None
_worker_hand_landmarker = None


def init_worker():
    """Initializes MediaPipe models ONCE per worker process."""
    global _worker_pose_landmarker, _worker_hand_landmarker
    silence_cpp_stderr()

    base_pose_opts = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH), delegate=python.BaseOptions.Delegate.CPU)
    pose_opts = vision.PoseLandmarkerOptions(base_options=base_pose_opts, running_mode=vision.RunningMode.IMAGE, min_pose_detection_confidence=0.5)
    _worker_pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

    base_hand_opts = python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH), delegate=python.BaseOptions.Delegate.CPU)
    hand_opts = vision.HandLandmarkerOptions(base_options=base_hand_opts, running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.4)
    _worker_hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)


def ensure_models_exist():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)
    if not POSE_MODEL_PATH.exists():
        print("📦 Downloading pose_landmarker model ...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    if not HAND_MODEL_PATH.exists():
        print("📦 Downloading hand_landmarker model ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)


def is_valid_landmark_file(file_path: Path) -> bool:
    if not file_path.exists() or file_path.stat().st_size < 1024:
        return False
    try:
        arr = np.load(str(file_path))
        if arr.ndim != 2 or arr.shape[1] != LM_DIM or arr.shape[0] < 4:
            return False
        if np.all(arr == 0):
            return False
        return True
    except Exception:
        if file_path.exists():
            try: file_path.unlink()
            except Exception: pass
        return False


def aug_speed(frames, factor):
    n = len(frames)
    new_n = max(4, int(n / factor))
    indices = np.linspace(0, n - 1, new_n).astype(int)
    return [frames[i] for i in indices]

def aug_brightness(frames, delta):
    out = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + delta), 0, 255)
        out.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    return out

def aug_contrast(frames, factor):
    out = []
    for f in frames:
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        mean_l = lab[:, :, 0].mean()
        lab[:, :, 0] = np.clip((lab[:, :, 0] - mean_l) * factor + mean_l, 0, 255)
        out.append(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR))
    return out

def aug_gaussian_noise(frames, sigma):
    out = []
    for f in frames:
        noise = np.random.normal(0, sigma, f.shape).astype(np.float32)
        out.append(np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return out

def aug_gaussian_blur(frames, ksize):
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return [cv2.GaussianBlur(f, (ksize, ksize), 0) for f in frames]

def aug_zoom(frames, factor):
    out = []
    h, w = frames[0].shape[:2]
    for f in frames:
        if factor > 1:
            crop_h, crop_w = int(h / factor), int(w / factor)
            y0, x0 = (h - crop_h) // 2, (w - crop_w) // 2
            out.append(cv2.resize(f[y0:y0 + crop_h, x0:x0 + crop_w], (w, h), interpolation=cv2.INTER_LINEAR))
        else:
            pad_h, pad_w = int(h * (1 - factor) / 2), int(w * (1 - factor) / 2)
            padded = cv2.copyMakeBorder(f, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
            out.append(cv2.resize(padded, (w, h), interpolation=cv2.INTER_LINEAR))
    return out

def aug_rotation(frames, angle_deg):
    h, w = frames[0].shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return [cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for f in frames]

def aug_translate(frames, tx_frac, ty_frac):
    h, w = frames[0].shape[:2]
    M = np.float32([[1, 0, int(w * tx_frac)], [0, 1, int(h * ty_frac)]])
    return [cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for f in frames]

def aug_drop_frames(frames, n_drop):
    if len(frames) <= n_drop + 4: return frames
    keep = sorted(random.sample(range(len(frames)), len(frames) - n_drop))
    return [frames[i] for i in keep]

def aug_duplicate_frames(frames, n_dup):
    res = list(frames)
    for _ in range(n_dup):
        idx = random.randint(0, len(res) - 1)
        res.insert(idx, res[idx])
    return res

AUG_POOL = [
    ("speed_0.60", aug_speed, {"factor": 0.60}), ("speed_0.75", aug_speed, {"factor": 0.75}),
    ("speed_1.25", aug_speed, {"factor": 1.25}), ("bright+0.20", aug_brightness, {"delta": 0.20}),
    ("bright-0.20", aug_brightness, {"delta": -0.20}), ("contrast+0.20", aug_contrast, {"factor": 1.20}),
    ("noise_light", aug_gaussian_noise, {"sigma": 8}), ("blur_slight", aug_gaussian_blur, {"ksize": 3}),
    ("zoom_1.10", aug_zoom, {"factor": 1.10}), ("rot_-5", aug_rotation, {"angle_deg": -5}),
    ("rot_+5", aug_rotation, {"angle_deg": 5}), ("trans_left", aug_translate, {"tx_frac": -0.05, "ty_frac": 0.0}),
    ("trans_right", aug_translate, {"tx_frac": 0.05, "ty_frac": 0.0}), ("drop_2", aug_drop_frames, {"n_drop": 2}),
    ("dup_2", aug_duplicate_frames, {"n_dup": 2}),
]

def generate_aug_sets(n=N_AUGMENTS, seed=RANDOM_SEED):
    rng = random.Random(seed)
    used, res = set(), []
    while len(res) < n:
        k = rng.choice([2, 3, 4])
        chosen = rng.sample(range(len(AUG_POOL)), k)
        fs = frozenset(chosen)
        if fs not in used:
            used.add(fs)
            res.append(chosen)
    return res


def resolve_video_path(file_path_str: str) -> Path | None:
    p = PROJECT_ROOT / file_path_str
    if p.exists() and p.stat().st_size > 1024: return p

    rel_data = p.relative_to(DATA_DIR) if DATA_DIR in p.parents else Path(file_path_str)
    for base in [CROPPED_DIR, PROCESSED_DIR, RAW_DIR, DATA_DIR]:
        candidate = base / rel_data
        if candidate.exists() and candidate.stat().st_size > 1024:
            return candidate
        candidate_flat = base / p.name
        if candidate_flat.exists() and candidate_flat.stat().st_size > 1024:
            return candidate_flat
    return None


def extract_landmarks(frame_seq):
    """Runs MediaPipe landmark inference using the persistent process instances."""
    global _worker_pose_landmarker, _worker_hand_landmarker
    rows = []
    for frame in frame_seq:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        row_vec = np.zeros(LM_DIM, dtype=np.float32)

        pose_res = _worker_pose_landmarker.detect(mp_image)
        if pose_res.pose_landmarks and len(pose_res.pose_landmarks[0]) >= POSE_LM:
            for idx, lm in enumerate(pose_res.pose_landmarks[0][:POSE_LM]):
                row_vec[idx * 3 : idx * 3 + 3] = [lm.x, lm.y, lm.z]

        hand_res = _worker_hand_landmarker.detect(mp_image)
        if hand_res.hand_landmarks and hand_res.handedness:
            for hand_lms, handedness_cat in zip(hand_res.hand_landmarks, hand_res.handedness):
                lbl = handedness_cat[0].category_name.lower()
                offset = 99 if lbl == "left" else 162
                for idx, lm in enumerate(hand_lms[:HAND_LM]):
                    row_vec[offset + idx * 3 : offset + idx * 3 + 3] = [lm.x, lm.y, lm.z]
        rows.append(row_vec)

    arr = np.stack(rows)
    for col in range(LM_DIM):
        zero_mask = (arr[:, col] == 0)
        if zero_mask.any() and not zero_mask.all():
            indices = np.arange(len(arr))
            arr[:, col] = np.interp(indices, indices[~zero_mask], arr[~zero_mask, col])
    return arr


def process_single_video_worker(row: dict, augment: bool) -> tuple[list[dict], str | None]:
    video_path = resolve_video_path(row["file_path"])
    label = row["label"]
    signer_id = row["signer_id"]
    identifier = f"{signer_id}/{label}"

    if not video_path or not video_path.exists():
        return [], f"❌ MISSING/CORRUPT VIDEO: {identifier} ({row['file_path']})"

    lm_base = (LANDMARK_DIR / signer_id / label)
    lm_base.mkdir(parents=True, exist_ok=True)
    orig_lm_path = lm_base / "original.npy"

    all_exist = is_valid_landmark_file(orig_lm_path)
    if augment and all_exist:
        for aug_idx in range(N_AUGMENTS):
            if not is_valid_landmark_file(lm_base / f"aug_{aug_idx:03d}.npy"):
                all_exist = False
                break

    if all_exist:
        out_rows = []
        row_orig = dict(row)
        row_orig["landmark_path"] = str(orig_lm_path.relative_to(PROJECT_ROOT))
        row_orig["augmented"] = False
        row_orig["aug_id"] = ""
        out_rows.append(row_orig)

        if augment:
            for aug_idx in range(N_AUGMENTS):
                aug_id = f"aug_{aug_idx:03d}"
                aug_row = dict(row)
                aug_row["landmark_path"] = str((lm_base / f"{aug_id}.npy").relative_to(PROJECT_ROOT))
                aug_row["augmented"] = True
                aug_row["aug_id"] = aug_id
                out_rows.append(aug_row)
        return out_rows, None

    # Pre-decode video to RAM
    try:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break
            if frame.shape[0] != 256 or frame.shape[1] != 256:
                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
    except Exception as e:
        return [], f"❌ OPENCV ERROR reading {identifier}: {e}"

    if not frames or len(frames) < 4:
        return [], f"⚠️ SKIPPED (Too few frames/corrupt): {identifier}"

    out_rows = []

    # Extract original landmarks
    if not is_valid_landmark_file(orig_lm_path):
        orig_lm = extract_landmarks(frames)
        np.save(str(orig_lm_path), orig_lm)

    row_orig = dict(row)
    row_orig["landmark_path"] = str(orig_lm_path.relative_to(PROJECT_ROOT))
    row_orig["augmented"] = False
    row_orig["aug_id"] = ""
    out_rows.append(row_orig)

    # Extract pixel-augmented landmarks
    if augment:
        vid_seed = RANDOM_SEED + abs(hash(row["file_path"])) % 100000
        aug_sets = generate_aug_sets(n=N_AUGMENTS, seed=vid_seed)

        for aug_idx, aug_combo in enumerate(aug_sets):
            aug_id = f"aug_{aug_idx:03d}"
            aug_lm_path = lm_base / f"{aug_id}.npy"

            if not is_valid_landmark_file(aug_lm_path):
                try:
                    aug_frames = frames
                    for idx in aug_combo:
                        _, fn, kwargs = AUG_POOL[idx]
                        aug_frames = fn(aug_frames, **kwargs)
                    aug_lm = extract_landmarks(aug_frames)
                    np.save(str(aug_lm_path), aug_lm)
                except Exception:
                    continue

            aug_row = dict(row)
            aug_row["landmark_path"] = str(aug_lm_path.relative_to(PROJECT_ROOT))
            aug_row["augmented"] = True
            aug_row["aug_id"] = aug_id
            out_rows.append(aug_row)

    del frames
    gc.collect()
    return out_rows, None


def process_manifest_parallel(manifest_file: str, augment: bool) -> None:
    csv_path = CSV_DIR / manifest_file
    if not csv_path.exists(): return

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("augmented", "").lower() != "true"]

    print(f"\n{'='*65}")
    print(f" 🚀 Processing: {manifest_file} ({len(rows)} Target Videos)")
    print(f"{'='*65}")

    # Physical core saturation: 7 single-threaded dedicated workers
    num_workers = 7

    final_rows = []
    error_logs = []
    total = len(rows)

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        futures = {executor.submit(process_single_video_worker, row, augment): row for row in rows}
        
        for idx, future in enumerate(as_completed(futures), 1):
            row_ref = futures[future]
            try:
                res_rows, err_msg = future.result()
                if err_msg:
                    print(f"  [{idx:03d}/{total:03d}] {err_msg}")
                    error_logs.append(err_msg)
                if res_rows:
                    final_rows.extend(res_rows)
                    if not err_msg:
                        now_str = datetime.now().strftime('%H:%M:%S')
                        print(f"  [{idx:03d}/{total:03d}] ✅ {res_rows[0]['signer_id']}/{res_rows[0]['label']} (+{len(res_rows)-1} augments) | Time: {now_str}")
            except Exception as exc:
                err = f"❌ WORKER EXCEPTION: {row_ref['signer_id']}/{row_ref['label']} -> {exc}"
                print(f"  [{idx:03d}/{total:03d}] {err}")
                error_logs.append(err)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        w.writerows(final_rows)

    if error_logs:
        error_log_file = LOG_DIR / f"04_missing_videos_{Path(manifest_file).stem}.txt"
        with open(error_log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(error_logs))

    print(f"  ✅ Saved {manifest_file} ({len(final_rows)} landmark entries)")


def main():
    ensure_models_exist()
    print("\n" + "═" * 65)
    print(" 🎯 SignPAK-AI High-Throughput Landmark Extraction Engine")
    print("═" * 65)
    process_manifest_parallel("manifest.csv", augment=True)
    process_manifest_parallel("train.csv",    augment=True)
    process_manifest_parallel("val.csv",      augment=False)
    process_manifest_parallel("test.csv",     augment=False)
    print("\n✅ All Landmark Extractions & Augmentations Complete!\n")


if __name__ == "__main__":
    main()