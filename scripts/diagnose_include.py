"""
diagnose_include.py — Nested Checkpoint Inspector
===================================================
Unwraps all nested dictionaries inside include_pretrained.pth
"""

import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "models" / "checkpoints" / "include_pretrained.pth"

def check():
    if not WEIGHTS_PATH.exists():
        print(f"❌ Checkpoint file not found at {WEIGHTS_PATH}")
        return

    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    print(f"📦 Raw Checkpoint Top-Level Type: {type(ckpt)}")

    if isinstance(ckpt, dict):
        print(f"Top-level keys found: {list(ckpt.keys())}")
        
        # Unwrap nested dicts if present
        for key in ["state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"🔍 Found inner dictionary under key: '{key}'")
                ckpt = ckpt[key]
                break

    print(f"\n✅ Total Tensors Extracted: {len(ckpt)}")
    print("=" * 60)
    print("First 10 Tensors inside Checkpoint:")
    print("=" * 60)

    for i, (k, v) in enumerate(ckpt.items()):
        if isinstance(v, torch.Tensor):
            print(f" - [{k:<35}] -> Shape: {tuple(v.shape)}")
        else:
            print(f" - [{k:<35}] -> Nested Type: {type(v)}")
        if i >= 15:
            break

if __name__ == "__main__":
    check()