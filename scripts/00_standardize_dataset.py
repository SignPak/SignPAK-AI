"""
00_standardize_dataset.py — SignPAK-AI (Directory & Filename Standardizer)
==========================================================================
Scans data/ (or data/raw/) and enforces exact folder and filename alignment 
matching Signer_0 specifications.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Folder Renaming Map (Plural -> Singular)
FOLDER_FIXES = {
    "English Alphabets": "English Alphabet",
    "Urdu Alphabets": "Urdu Alphabet"
}

# File Renaming Map (Typos / Casing)
FILE_FIXES = {
    "Saud.mp4": "Suad.mp4",
    "one.mp4": "One.mp4",
}


def standardize_directory_tree():
    print("=" * 60)
    print("🔧 STANDARDIZING SIGNPAK-AI FOLDERS & FILENAMES")
    print("=" * 60)

    # 1. Search across data/ and data/raw/
    candidate_roots = [DATA_DIR, DATA_DIR / "raw", Path(".")]
    signer_dirs = []

    for root in candidate_roots:
        if root.exists():
            for p in root.iterdir():
                if p.is_dir() and p.name.startswith("Signer_") and p not in signer_dirs:
                    signer_dirs.append(p)

    if not signer_dirs:
        print("❌ No 'Signer_*' folders found to standardize.")
        return

    # 2. Fix subfolder names
    for signer in signer_dirs:
        for old_name, new_name in FOLDER_FIXES.items():
            old_path = signer / old_name
            new_path = signer / new_name
            if old_path.exists():
                print(f"📁 Renaming Folder: {signer.name}/{old_name} ──> {new_name}")
                old_path.rename(new_path)

    # 3. Fix file names & typos
    for signer in signer_dirs:
        for old_file, new_file in FILE_FIXES.items():
            for found_path in signer.rglob(old_file):
                target_path = found_path.parent / new_file
                print(f"📄 Renaming File: {signer.name}/{found_path.parent.name}/{old_file} ──> {new_file}")
                found_path.rename(target_path)

    # 4. Check for duplicate take files like Qaaf_2.mp4
    for signer in signer_dirs:
        for dup in signer.rglob("*_2.mp4"):
            print(f"⚠️ Notice duplicate file: {signer.name}/{dup.parent.name}/{dup.name}")
            print(f"   (If this is an alternate recording, keep the better one as {dup.stem.split('_')[0]}.mp4)")

    print("\n" + "=" * 60)
    print("✅ DIRECTORY STANDARDIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    standardize_directory_tree()