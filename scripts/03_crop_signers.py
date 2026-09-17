"""
03_crop_signers.py — SignPAK-AI (Dynamic Scale Bounding Box Engine)
===================================================================
- Dynamically crops all Signer directories in data/processed/.
- Crops upper-body bounding boxes to 256x256 resolution.
- Auto-skips previously cropped videos in data/cropped/.

Input  : data/processed/**/*.mp4
Output : data/cropped/**/*.mp4
"""

import cv2
import json
import urllib.request
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IN_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR / "cropped"
LOG_DIR = DATA_DIR / "logs"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "pose_landmarker.task"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

OUTPUT_SIZE = (256, 256)
PADDING_FRAC = 0.20
UPPER_BODY_LM = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


def ensure_model_exists():
    if not MODEL_PATH.exists():
        print(f"📦 Downloading pose_landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Download complete!\n")

ensure_model_exists()

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH),
    delegate=python.BaseOptions.Delegate.CPU
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5
)
landmarker = vision.PoseLandmarker.create_from_options(options)


def get_pose_bbox(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    try:
        result = landmarker.detect(mp_image)
    except Exception:
        return None

    if not result.pose_landmarks:
        return None

    lms = result.pose_landmarks[0]
    xs, ys = [], []

    for idx in UPPER_BODY_LM:
        if idx < len(lms):
            lm = lms[idx]
            visibility = getattr(lm, 'visibility', 1.0)
            if visibility >= 0.3:
                xs.append(lm.x * w)
                ys.append(lm.y * h)

    if not xs:
        return None

    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))

    pad_x = int((x2 - x1) * PADDING_FRAC)
    pad_y = int((y2 - y1) * PADDING_FRAC)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return x1, y1, x2, y2


def median_bbox(bboxes: list[tuple]) -> tuple[int, int, int, int]:
    arr = np.array(bboxes)
    return tuple(int(np.median(arr[:, i])) for i in range(4))


def _resize_only(in_path: Path, out_path: Path, fps: float) -> dict:
    cap = cv2.VideoCapture(str(in_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, OUTPUT_SIZE)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(cv2.resize(frame, OUTPUT_SIZE, interpolation=cv2.INTER_AREA))
    cap.release()
    writer.release()
    return {"status": "fallback_resize_only", "bbox_median": None}


def crop_video(in_path: Path, out_path: Path) -> dict:
    cap = cv2.VideoCapture(str(in_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total_frames <= 0:
        cap.release()
        return {"status": "invalid_or_corrupt_video", "bbox_median": None}

    sample_indices = np.linspace(0, total_frames - 1, min(10, total_frames), dtype=int).tolist()
    bboxes = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        bbox = get_pose_bbox(frame)
        if bbox:
            bboxes.append(bbox)

    if not bboxes:
        cap.release()
        return _resize_only(in_path, out_path, fps)

    x1, y1, x2, y2 = median_bbox(bboxes)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, OUTPUT_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = get_pose_bbox(frame)
        fx1, fy1, fx2, fy2 = bbox if bbox else (x1, y1, x2, y2)

        cropped = frame[fy1:fy2, fx1:fx2]
        if cropped.size == 0:
            cropped = frame

        resized = cv2.resize(cropped, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
        writer.write(resized)

    cap.release()
    writer.release()

    return {
        "status": "ok",
        "bbox_median": [x1, y1, x2, y2],
        "frames_sampled_for_bbox": len(bboxes),
    }


def process_all() -> None:
    all_videos = sorted(IN_DIR.rglob("*.mp4"))

    if not all_videos:
        print(f"❌ No videos found in {IN_DIR}")
        print("   Run 02_segment_videos.py first.")
        return

    print(f"Found {len(all_videos)} processed videos across ALL signers to crop.\n")

    log_entries = []
    stats = defaultdict(int)

    for mp4 in all_videos:
        rel = mp4.relative_to(IN_DIR)
        out = OUT_DIR / rel

        if out.exists() and out.stat().st_size > 1000:
            print(f"  ⏩ {rel} (Already cropped, skipping)")
            stats["ok"] += 1
            continue

        print(f"  {rel} ...", end=" ", flush=True)

        try:
            result = crop_video(mp4, out)
            status = result["status"]
            if status == "ok":
                print(f"✅ bbox {result['bbox_median']}")
                stats["ok"] += 1
            else:
                print(f"⚠️ {status}")
                stats["fallback"] += 1
        except Exception as exc:
            print(f"❌ {exc}")
            result = {"status": f"error: {exc}"}
            stats["error"] += 1

        log_entries.append({
            "source": str(mp4.relative_to(PROJECT_ROOT)),
            "output": str(out.relative_to(PROJECT_ROOT)),
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            **result,
        })

    log_path = LOG_DIR / "03_crop_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2)

    print("\n" + "=" * 60)
    print("  CROP SUMMARY")
    print("=" * 60)
    print(f"   Cropped      : {stats['ok']}")
    print(f"   Resize only  : {stats['fallback']}")
    print(f"   Errors       : {stats['error']}")
    print(f"   Output dir   : {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    process_all()