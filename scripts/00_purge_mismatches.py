"""
00_purge_mismatches.py — SignPAK-AI (Mismatch Auditor & Purger)
================================================================
Scans data/Signer_0/ and data/metadata/ for mismatched compound videos
(e.g., 'Single Bed.mp4' or 'Third Position.mp4' saved for single-word targets).
"""

import json
import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
WORDS_LIST_DIR = DATA_DIR / "Words List"
SIGNER_0_DIR = DATA_DIR / "Signer_0"


def normalize(text: str) -> str:
    """Strips parenthetical text, special characters, and extra spaces."""
    t = re.sub(r'\(.*?\)', '', str(text))
    t = re.sub(r'[^\w\s]', ' ', t)
    return " ".join(t.lower().split())


def find_master_file() -> Path | None:
    """Locates master_dataset.json regardless of exact subfolder structure."""
    candidates = list(DATA_DIR.rglob("master_dataset.json"))
    if candidates:
        return candidates[0]
    return None


def audit_and_purge():
    # 1. Load targets from CSV
    csv_files = sorted(WORDS_LIST_DIR.glob("Words - *.csv"))
    if not csv_files:
        csv_files = sorted(Path(".").glob("Words - *.csv"))
    
    target_set = set()
    for csv_f in csv_files:
        df = pd.read_csv(csv_f)
        for _, row in df.iterrows():
            w = str(row.get("English Word", "")).strip()
            if w and w.lower() != "nan":
                target_set.add(normalize(w))

    print(f"📋 Loaded {len(target_set)} unique target words from CSV.")

    master_file = find_master_file()
    purged_count = 0

    # Option A: Audit via master_dataset.json if available
    if master_file and master_file.exists():
        print(f"🔍 Auditing via master file: {master_file.relative_to(PROJECT_ROOT)}")
        with open(master_file, "r", encoding="utf-8") as f:
            master_data = json.load(f)

        clean_master = []
        for cat_entry in master_data:
            if not isinstance(cat_entry, dict) or "concepts" not in cat_entry:
                continue

            valid_concepts = []
            for concept in cat_entry.get("concepts", []):
                psl_title = str(concept.get("title", "")).strip()
                norm_psl = normalize(psl_title)
                video_path_str = concept.get("video_path", "")
                video_path = Path(video_path_str) if video_path_str else None

                if video_path and video_path.exists():
                    file_stem_norm = normalize(video_path.stem)

                    # Check if PSL title is a compound phrase matching a single-word target
                    if norm_psl != file_stem_norm and file_stem_norm in target_set:
                        print(f"  ❌ MISMATCH DETECTED: File '{video_path.name}' came from PSL phrase '{psl_title}'")
                        try:
                            video_path.unlink()
                            print(f"     🗑️ Deleted incorrect file: {video_path}")
                            purged_count += 1
                            continue
                        except Exception as e:
                            print(f"     ⚠️ Could not delete {video_path}: {e}")

                valid_concepts.append(concept)

            cat_entry["concepts"] = valid_concepts
            clean_master.append(cat_entry)

        with open(master_file, "w", encoding="utf-8") as f:
            json.dump(clean_master, f, ensure_ascii=False, indent=2)

    # Option B: Direct Disk Scan of Signer_0 for known compound phrase filenames
    else:
        print("ℹ️ master_dataset.json not found. Scanning Signer_0 disk files directly...")
        if SIGNER_0_DIR.exists():
            for mp4 in SIGNER_0_DIR.rglob("*.mp4"):
                stem_norm = normalize(mp4.stem)
                # Identify files where stem doesn't match any target word exactly
                if stem_norm not in target_set:
                    print(f"  ❌ UNTARGETED / COMPOUND FILE DETECTED: '{mp4.name}'")
                    try:
                        mp4.unlink()
                        print(f"     🗑️ Deleted: {mp4}")
                        purged_count += 1
                    except Exception as e:
                        print(f"     ⚠️ Could not delete {mp4}: {e}")

    print("\n" + "=" * 60)
    print(" ✅ AUDIT & PURGE COMPLETE!")
    print("=" * 60)
    print(f"  • Mismatched Videos Deleted : {purged_count}")
    print("  • Correct Videos Retained  : Safe on disk")
    print("=" * 60)


if __name__ == "__main__":
    audit_and_purge()