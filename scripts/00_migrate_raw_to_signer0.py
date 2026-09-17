"""
00_migrate_raw_to_signer0.py — SignPAK-AI
==========================================
Migrates all category folders and videos from data/raw to data/Signer_0.
"""

import shutil
from pathlib import Path

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
SIGNER0_DIR = DATA_DIR / "Signer_0"


def migrate():
    if not RAW_DIR.exists():
        print("ℹ️ No 'data/raw' directory found. Migration skipped or already complete.")
        return

    SIGNER0_DIR.mkdir(parents=True, exist_ok=True)
    moved_count = 0

    for item in RAW_DIR.rglob("*.mp4"):
        rel_path = item.relative_to(RAW_DIR)
        target_path = SIGNER0_DIR / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not target_path.exists():
            shutil.move(str(item), str(target_path))
            moved_count += 1
        else:
            # If video already exists, replace cleanly
            shutil.copy2(str(item), str(target_path))
            item.unlink()
            moved_count += 1

    print(f"✅ Migrated {moved_count} video files from 'data/raw' -> 'data/Signer_0'.")

    try:
        shutil.rmtree(RAW_DIR)
        print("🗑️ Removed legacy 'data/raw' directory.")
    except Exception as e:
        print(f"⚠️ Note: Could not remove empty raw folder: {e}")


if __name__ == "__main__":
    migrate()