"""Central configuration. Thresholds are config, never hard-coded elsewhere."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = Path(os.getenv("FACEGUARD_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.getenv("FACEGUARD_DB", DATA_DIR / "faceguard.db"))
MODEL_DIR = Path(os.getenv("FACEGUARD_MODEL_DIR", DATA_DIR / "models"))


class Settings(BaseModel):
    # Decision thresholds (calibrate with scripts/evaluate.py)
    t_sim: float = float(os.getenv("FACEGUARD_T_SIM", 0.35))
    t_fake: float = float(os.getenv("FACEGUARD_T_FAKE", 0.5))

    # Models
    insightface_pack: str = "buffalo_l"
    det_size: int = 640
    detector_model_id: str = os.getenv(
        "FACEGUARD_DETECTOR", "prithivMLmods/Deep-Fake-Detector-v2-Model"
    )
    detector_fallback_id: str = "dima806/deepfake_vs_real_image_detection"

    # Preprocessing
    min_face_px: int = 60
    fake_crop_margin: float = 0.30
    fake_input_size: int = 224
    enroll_min_images: int = 1  # API accepts 1+, UI asks for 3-5
    enroll_max_images: int = 10


settings = Settings()
