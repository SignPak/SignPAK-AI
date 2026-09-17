"""
02_segment_videos.py — SignPAK-AI (Universal Dynamic Video Segmenter)
=====================================================================
- Automatically discovers all data/raw/Signer_* or data/Signer_* folders.
- Preserves full category hierarchy dynamically without hardcoded maps.
- Auto-skips previously processed videos in data/processed/.

Input  : data/raw/Signer_*/**/*.mp4 OR data/Signer_*/**/*.mp4
Output : data/processed/<Signer_X>/<Category>/<label>.mp4
"""

import re
import cv2
import json
import shutil
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
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
LOG_DIR = DATA_DIR / "logs"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

LOG_DIR.mkdir(parents=True, exist_ok=True)

MIN_SIGN_FRAMES = 8
SILENCE_THRESH = 10
PAD_FRAMES = 3
HAND_CONFIDENCE = 0.5
BOTTOM_CUTOFF_RATIO = 0.85

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Hand landmarker model missing at: {MODEL_PATH}")

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=HAND_CONFIDENCE,
    min_tracking_confidence=0.4
)
landmarker = vision.HandLandmarker.create_from_options(options)


def normalize_label(stem: str) -> str:
    t = re.sub(r'\(.*?\)', '', str(stem))
    t = re.sub(r'[^\w\s]', ' ', t)
    return "_".join(t.lower().split())


def get_hand_presence(video_path: Path) -> list[bool]:
    cap = cv2.VideoCapture(str(video_path))
    presence = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        valid = False
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                if hand[0].y < BOTTOM_CUTOFF_RATIO:
                    valid = True
                    break
        presence.append(valid)
    cap.release()
    return presence


def find_signing_windows(presence: list[bool]) -> list[tuple[int, int]]:
    windows = []
    n = len(presence)
    i = 0
    while i < n:
        if not presence[i]:
            i += 1
            continue
        start = i
        silent_count = 0
        j = i
        while j < n:
            if presence[j]:
                silent_count = 0
            else:
                silent_count += 1
                if silent_count >= SILENCE_THRESH:
                    break
            j += 1
        end = j - silent_count
        if (end - start) >= MIN_SIGN_FRAMES:
            windows.append((start, end))
        i = j + 1
    return windows


def extract_clip(video_path: Path, start: int, end: int, out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    s = max(0, start - PAD_FRAMES)
    e = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), end + PAD_FRAMES)

    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    for _ in range(s, e):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    return out_path.exists() and out_path.stat().st_size > 1000


def find_all_signer_dirs() -> list[Path]:
    signer_dirs = []
    for root in [RAW_DIR, DATA_DIR]:
        if root.exists():
            for p in sorted(root.iterdir()):
                if p.is_dir() and re.match(r"^Signer_\d+$", p.name, re.IGNORECASE):
                    if p not in signer_dirs:
                        signer_dirs.append(p)
    return signer_dirs


def process_video_sources():
    signer_dirs = find_all_signer_dirs()

    if not signer_dirs:
        print("❌ No 'Signer_*' directories found in data/raw/ or data/")
        return

    log_entries = []
    stats = defaultdict(int)

    for target_dir in signer_dirs:
        source_name = target_dir.name
        print(f"\n── Processing Directory: {source_name} ────────────────────")
        videos = sorted(list(target_dir.rglob("*.mp4")) + list(target_dir.rglob("*.avi")))

        for mp4 in videos:
            # Preserve original subcategory dynamically
            category = mp4.parent.name
            label = normalize_label(mp4.stem)
            out_path = OUT_DIR / source_name / category / f"{label}.mp4"

            if out_path.exists() and out_path.stat().st_size > 1000:
                print(f"  ⏩ {source_name}/{category}/{label} (Already segmented, skipping)")
                stats["ok"] += 1
                continue

            entry = {
                "source_dir": source_name,
                "category": category,
                "label": label,
                "source": str(mp4.relative_to(PROJECT_ROOT)),
                "output": str(out_path.relative_to(PROJECT_ROOT)),
                "status": "",
                "windows_found": 0,
                "window_used": "",
                "processed_at": datetime.now().isoformat(timespec="seconds"),
            }

            print(f"  {source_name}/{category}/{label} ...", end=" ", flush=True)

            try:
                presence = get_hand_presence(mp4)
            except Exception as exc:
                print(f"❌ (hand detection failed: {exc})")
                entry["status"] = f"error: {exc}"
                log_entries.append(entry)
                stats["error"] += 1
                continue

            windows = find_signing_windows(presence)
            entry["windows_found"] = len(windows)

            if not windows:
                print("⚠️ no sign window detected, copying full video")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mp4, out_path)
                entry["status"] = "fallback_full_copy"
                entry["window_used"] = f"0-{len(presence)}"
                stats["fallback"] += 1
            else:
                start, end = windows[0]
                entry["window_used"] = f"{start}-{end}"
                ok = extract_clip(mp4, start, end, out_path)
                if ok:
                    fps = cv2.VideoCapture(str(mp4)).get(cv2.CAP_PROP_FPS) or 25.0
                    dur_s = round((end - start) / max(fps, 1.0), 2)
                    print(f"✅ frames {start}–{end} ({dur_s}s)")
                    entry["status"] = "ok"
                    stats["ok"] += 1
                else:
                    print("❌ write failed")
                    entry["status"] = "write_failed"
                    stats["error"] += 1

            log_entries.append(entry)

    log_path = LOG_DIR / "02_segmentation_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(" PROCESSING COMPLETE")
    print("=" * 60)
    print(f"   Trimmed & Cleaned : {stats['ok']}")
    print(f"   Fallback Copied   : {stats['fallback']}")
    print(f"   Errors            : {stats['error']}")
    print("=" * 60)


if __name__ == "__main__":
    process_video_sources()