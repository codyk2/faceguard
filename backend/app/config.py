"""Central configuration. Thresholds are config, never hard-coded elsewhere."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = Path(os.getenv("FACEGUARD_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.getenv("FACEGUARD_DB", DATA_DIR / "faceguard.db"))
MODEL_DIR = Path(os.getenv("FACEGUARD_MODEL_DIR", DATA_DIR / "models"))


BASE_DETECTOR = "prithivMLmods/Deep-Fake-Detector-v2-Model"
# Fine-tuned on DeepFakeFace by scripts/train_detector.py (--degrade strong). Not in git;
# rebuild with `make train-detector`. Falls back to the base model if absent.
FINETUNED_DETECTOR = MODEL_DIR / "detector-dff-ft-lowres"
_HAVE_FT = (FINETUNED_DETECTOR / "config.json").exists()


class Settings(BaseModel):
    # Decision thresholds. T_FAKE for the fine-tuned detector is calibrated on held-out LFW
    # real photos to ~5% false positives (scripts/train_detector.py calibrate); the base
    # model is uncalibrated, so it keeps 0.5.
    t_sim: float = float(os.getenv("FACEGUARD_T_SIM", 0.35))
    t_fake: float = float(os.getenv("FACEGUARD_T_FAKE", 0.90 if _HAVE_FT else 0.5))

    # Models
    insightface_pack: str = "buffalo_l"
    det_size: int = 640
    detector_model_id: str = os.getenv(
        "FACEGUARD_DETECTOR", str(FINETUNED_DETECTOR) if _HAVE_FT else BASE_DETECTOR
    )
    detector_fallback_id: str = "dima806/deepfake_vs_real_image_detection"

    # Preprocessing
    min_face_px: int = 60
    fake_crop_margin: float = 0.30
    fake_input_size: int = 224
    enroll_min_images: int = 1  # API accepts 1+, UI asks for 3-5
    enroll_max_images: int = 10


settings = Settings()
