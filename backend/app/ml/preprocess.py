"""Image decoding and the single face-detection pass shared by both models.

Flow: bytes -> BGR ndarray -> SCRFD (InsightFace buffalo_l) -> largest face ->
  (a) 5-point aligned 112x112 crop for ArcFace
  (b) bbox crop expanded by `settings.fake_crop_margin`, as RGB PIL, for the ViT
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # insightface imports albumentations

import cv2
import numpy as np
from PIL import Image

from app.config import MODEL_DIR, Settings, settings as default_settings
from app.ml.interfaces import NoFaceError
from app.schemas import FaceBox

MULTI_FACE_WARNING = "multiple faces detected; using the largest, most central one"
INSIGHTFACE_PROVIDERS = ["CPUExecutionProvider"]  # CoreML is flaky with these ONNX graphs


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode encoded image bytes (jpg/png/...) into a BGR uint8 array."""
    if not image_bytes:
        raise ValueError("empty image")
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("could not decode image")
    return img


def expand_box(
    box: tuple[float, float, float, float], margin: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Grow a box so each side is (1 + margin) times longer, centred, clipped to the image."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * margin / 2.0, h * margin / 2.0
    nx1 = int(max(0, np.floor(x1 - dx)))
    ny1 = int(max(0, np.floor(y1 - dy)))
    nx2 = int(min(width, np.ceil(x2 + dx)))
    ny2 = int(min(height, np.ceil(y2 + dy)))
    return nx1, ny1, nx2, ny2


@dataclass
class DetectedFace:
    face_box: FaceBox
    kps: np.ndarray                 # (5, 2) landmarks
    aligned_crop: np.ndarray        # (112, 112, 3) BGR, ArcFace input
    fake_crop: Image.Image          # RGB margin crop, ViT input
    warnings: list[str] = field(default_factory=list)


class FacePreprocessor:
    """Owns the InsightFace model pack. Detection runs once per image."""

    def __init__(self, cfg: Settings | None = None, providers: list[str] | None = None):
        from insightface.app import FaceAnalysis

        self.cfg = cfg or default_settings
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.app = FaceAnalysis(
            name=self.cfg.insightface_pack,
            root=str(MODEL_DIR),
            providers=providers or INSIGHTFACE_PROVIDERS,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=-1, det_size=(self.cfg.det_size, self.cfg.det_size))
        self.det_model = self.app.det_model
        self.rec_model = self.app.models["recognition"]
        self.device = "cpu"

    def detect(self, img_bgr: np.ndarray) -> DetectedFace:
        from insightface.utils import face_align

        bboxes, kpss = self.det_model.detect(img_bgr, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            raise NoFaceError("no face detected")

        # Pick the main face: visible (image-clipped) area, weighted by closeness to the
        # image centre. Raw SCRFD boxes for faces cut off at the edge extend past the
        # image and would otherwise beat the centred subject (seen on LFW crops).
        h, w = img_bgr.shape[:2]
        cx1 = np.clip(bboxes[:, 0], 0, w)
        cy1 = np.clip(bboxes[:, 1], 0, h)
        cx2 = np.clip(bboxes[:, 2], 0, w)
        cy2 = np.clip(bboxes[:, 3], 0, h)
        areas = (cx2 - cx1) * (cy2 - cy1)
        dist = np.hypot((cx1 + cx2) / 2 - w / 2, (cy1 + cy2) / 2 - h / 2)
        centre_weight = 1.0 - 0.5 * dist / (np.hypot(w, h) / 2)
        i = int(np.argmax(areas * centre_weight))
        x1, y1, x2, y2, score = (float(v) for v in bboxes[i])
        warnings = [MULTI_FACE_WARNING] if len(bboxes) > 1 else []

        short_side = min(x2 - x1, y2 - y1)
        if short_side < self.cfg.min_face_px:
            raise NoFaceError(
                f"face too small ({short_side:.0f}px short side, need {self.cfg.min_face_px}px)"
            )

        kps = kpss[i]
        aligned = face_align.norm_crop(img_bgr, landmark=kps, image_size=112)

        h, w = img_bgr.shape[:2]
        cx1, cy1, cx2, cy2 = expand_box((x1, y1, x2, y2), self.cfg.fake_crop_margin, w, h)
        crop = cv2.cvtColor(img_bgr[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)

        return DetectedFace(
            face_box=FaceBox(
                x1=max(0.0, x1), y1=max(0.0, y1), x2=min(float(w), x2), y2=min(float(h), y2),
                score=score,
            ),
            kps=kps,
            aligned_crop=aligned,
            fake_crop=Image.fromarray(crop),
            warnings=warnings,
        )
