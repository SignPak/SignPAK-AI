"""
01_metadata.py — SignPAK-AI (Universal Dataset Manifest Indexer)
================================================================
- Dynamically indexes N signers and M classes across all CSVs.
- Auto-detects data/raw/Signer_* or data/Signer_*.
- Outputs data/csv/manifest.csv, labels.csv, and train/val/test splits.

Run from: SIGNPAK-AI root → python scripts/01_metadata.py
"""

import os
import csv
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WORDS_LIST_DIR = DATA_DIR / "Words List"
CSV_DIR = DATA_DIR / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)


def normalize_title(text: str) -> str:
    t = re.sub(r'\(.*?\)', '', str(text))
    t = re.sub(r'[^\w\s]', ' ', t)
    return " ".join(t.lower().split())


def load_target_label_map() -> dict:
    valid_labels = {}
    csv_files = sorted(WORDS_LIST_DIR.glob("Words - *.csv"))
    if not csv_files:
        csv_files = sorted(DATA_DIR.glob("Words - *.csv")) + sorted(Path(".").glob("Words - *.csv"))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                eng_word = str(row.get("English Word", "")).strip()
                cat = str(row.get("Category", "")).strip()
                if eng_word and eng_word.lower() != "nan":
                    clean_label = normalize_title(eng_word).replace(" ", "_")
                    valid_labels[clean_label] = {
                        "english_word": eng_word,
                        "category": cat
                    }
        except Exception as e:
            print(f"⚠️ Could not load CSV {csv_file.name}: {e}")

    return valid_labels


def find_all_signer_dirs() -> list[Path]:
    signer_dirs = []
    # Check data/raw/ first, fallback to data/
    search_roots = [RAW_DIR, DATA_DIR]
    for root in search_roots:
        if root.exists():
            for p in sorted(root.iterdir()):
                if p.is_dir() and re.match(r"^Signer_\d+$", p.name, re.IGNORECASE):
                    if p not in signer_dirs:
                        signer_dirs.append(p)
    return signer_dirs


def generate_metadata_manifests():
    print("🔍 Scanning dataset directories for all signers and vocabulary...")
    label_info_map = load_target_label_map()
    print(f"📋 Target vocabulary entries loaded: {len(label_info_map)}")

    signer_dirs = find_all_signer_dirs()
    print(f"👥 Discovered {len(signer_dirs)} Signer directories: {[d.name for d in signer_dirs]}")

    if not signer_dirs:
        print("❌ No 'Signer_*' folders found in data/raw/ or data/")
        return

    all_samples = []

    for signer_dir in signer_dirs:
        signer_id = signer_dir.name
        video_files = list(signer_dir.rglob("*.mp4")) + list(signer_dir.rglob("*.avi"))

        for vfile in video_files:
            raw_word = vfile.stem
            clean_word_label = normalize_title(raw_word).replace(" ", "_")

            if not label_info_map or clean_word_label in label_info_map:
                category = vfile.parent.name
                rel_path = vfile.relative_to(PROJECT_ROOT)

                all_samples.append({
                    "signer_id": signer_id,
                    "category": category,
                    "label": clean_word_label,
                    "is_extra": False,
                    "file_path": str(rel_path),
                    "landmark_path": f"data/landmarks/{signer_id}/{clean_word_label}/original.npy",
                    "augmented": False,
                    "aug_id": ""
                })

    print(f"📦 Total Videos Discovered Across All Signers: {len(all_samples)}")

    if not all_samples:
        print("⚠️ No matching video samples found. Please check folder contents.")
        return

    # Master Labels Mapping
    unique_labels = sorted(list(set(s["label"] for s in all_samples)))
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}

    with open(CSV_DIR / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "label_idx"])
        for lbl, idx in label_map.items():
            writer.writerow([lbl, idx])

    for s in all_samples:
        s["label_idx"] = label_map[s["label"]]

    df_all = pd.DataFrame(all_samples)
    
    # Save full unaugmented manifest (Used by LOSO Cross-Validation)
    cols = ["signer_id", "split", "label", "label_idx", "category", "is_extra", "file_path", "landmark_path", "augmented", "aug_id"]
    df_all["split"] = "all"
    df_all[cols].to_csv(CSV_DIR / "manifest.csv", index=False)

    # Stratified Splits (For standard non-LOSO workflows)
    try:
        train_df, test_val_df = train_test_split(
            df_all, test_size=0.30, random_state=42,
            stratify=df_all["label"] if df_all["label"].value_counts().min() >= 2 else None
        )
        val_df, test_df = train_test_split(
            test_val_df, test_size=0.50, random_state=42,
            stratify=test_val_df["label"] if test_val_df["label"].value_counts().min() >= 2 else None
        )
    except Exception:
        train_df, test_val_df = train_test_split(df_all, test_size=0.30, random_state=42)
        val_df, test_df = train_test_split(test_val_df, test_size=0.50, random_state=42)

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "val"
    test_df = test_df.copy(); test_df["split"] = "test"

    train_df[cols].to_csv(CSV_DIR / "train.csv", index=False)
    val_df[cols].to_csv(CSV_DIR / "val.csv", index=False)
    test_df[cols].to_csv(CSV_DIR / "test.csv", index=False)

    print("\n" + "=" * 60)
    print(" ✅ METADATA MANIFESTS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"  • Total Unique Classes Indexed : {len(unique_labels)}")
    print(f"  • Active Signers               : {len(signer_dirs)}")
    print(f"  • Master Manifest Rows         : {len(df_all)}")
    print(f"  • Master File Path             : {CSV_DIR / 'manifest.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    generate_metadata_manifests()