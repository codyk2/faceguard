"""ArcFace (InsightFace buffalo_l, ResNet50 w600k) embeddings and cosine similarity."""
from __future__ import annotations

import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        raise ValueError("cannot normalise a zero or non-finite vector")
    return (v / n).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; inputs need not be pre-normalised."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


class ArcFaceVerifier:
    """Embeds a 5-point aligned 112x112 BGR crop with InsightFace's ArcFace ONNX model."""

    name = "InsightFace buffalo_l ArcFace (ResNet50, 512-d)"

    def __init__(self, rec_model, device: str = "cpu"):
        self.rec_model = rec_model
        self.device = device

    def embed(self, aligned_crop_bgr: np.ndarray) -> np.ndarray:
        """Return a (512,) L2-normalised float32 embedding."""
        feat = self.rec_model.get_feat(aligned_crop_bgr)  # same path as Face.normed_embedding
        return l2_normalize(np.asarray(feat).flatten())
