"""Shared helpers for the data / eval scripts.

Importing this module puts backend/ on sys.path so `import app...` works when a
script is run as `cd backend && uv run python ../scripts/<name>.py`.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def data_dir() -> Path:
    """Same resolution rule as app.config.DATA_DIR, without importing pydantic."""
    return Path(os.getenv("FACEGUARD_DATA_DIR", ROOT / "data"))


DATA_DIR = data_dir()
LFW_DIR = DATA_DIR / "lfw"
FAKES_DIR = DATA_DIR / "fakes"
EVAL_DIR = DATA_DIR / "eval"
RAW_DIR = DATA_DIR / "raw"
SEED_MANIFEST = DATA_DIR / "seed_manifest.json"

SEED = 1234


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def lfw_people(min_images: int = 1) -> dict[str, list[Path]]:
    """{person: sorted image paths} for the extracted LFW tree."""
    out: dict[str, list[Path]] = {}
    if not LFW_DIR.is_dir():
        return out
    for d in sorted(p for p in LFW_DIR.iterdir() if p.is_dir()):
        imgs = sorted(d.glob("*.jpg"))
        if len(imgs) >= min_images:
            out[d.name] = imgs
    return out


def lfw_image_path(name: str, idx: int) -> Path:
    """LFW 1-based image index -> path (pairs.txt convention)."""
    return LFW_DIR / name / f"{name}_{idx:04d}.jpg"
