"""
00_sanitize_metadata.py — SignPAK-AI (Metadata Sanitizer)
==========================================================
1. Replaces all legacy 'data/raw/' strings in JSON metadata with 'data/Signer_0/'.
2. Removes duplicate/overlapping concept entries sharing identical video paths.

Run from: SIGNPAK-AI root → python scripts/00_sanitize_metadata.py
"""

import json
from pathlib import Path

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
META_DIR = PROJECT_ROOT / "data" / "metadata"


def sanitize_json_files():
    if not META_DIR.exists():
        print("ℹ️ No metadata directory found.")
        return

    json_files = sorted(META_DIR.rglob("*.json"))
    print(f"🔍 Found {len(json_files)} metadata files to sanitize...\n")

    for jpath in json_files:
        rel_path = jpath.relative_to(PROJECT_ROOT)
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                content = f.read()

            # Fix legacy raw paths
            if "data\\raw" in content or "data/raw" in content:
                content = content.replace("data\\\\raw", "data\\\\Signer_0").replace("data/raw", "data/Signer_0")

            data = json.loads(content)

            # Deduplicate video_path mappings in master_dataset.json
            if jpath.name == "master_dataset.json" and isinstance(data, list):
                seen_paths = set()
                for cat in data:
                    if isinstance(cat, dict) and "concepts" in cat:
                        clean_concepts = []
                        for concept in cat["concepts"]:
                            vpath = concept.get("video_path", "")
                            if vpath and vpath in seen_paths:
                                print(f"  ⚠️ Removed duplicate mapping: '{concept.get('title')}' sharing path: {vpath}")
                                continue
                            if vpath:
                                seen_paths.add(vpath)
                            clean_concepts.append(concept)
                        cat["concepts"] = clean_concepts

            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"  ✅ Sanitized: {rel_path}")

        except Exception as e:
            print(f"  ❌ Error processing {rel_path}: {e}")

    print("\n" + "=" * 60)
    print(" ✅ METADATA SANITIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    sanitize_json_files()