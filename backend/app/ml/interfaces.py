"""Internal contract between ML core (ml/) and the API layer (api/, main.py).

The API layer depends ONLY on the names in this file. ml/pipeline.py provides
`load_pipeline() -> Pipeline`; tests use a fake that satisfies the same Protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from app.schemas import Decision, FaceBox


class NoFaceError(Exception):
    """Raised when no usable face is found (none detected, or too small)."""


@dataclass
class FaceAnalysis:
    embedding: np.ndarray          # shape (512,), L2-normalised float32
    fake_probability: float        # P(fake) in [0, 1]
    fake_label: str                # label name from model.config.id2label
    face_box: FaceBox
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class VerifyResult:
    decision: Decision
    similarity: float | None
    fake_probability: float | None
    face_box: FaceBox | None
    warnings: list[str]
    timings_ms: dict[str, float]


class Pipeline(Protocol):
    device: str
    verifier_name: str
    detector_name: str

    def analyze(self, image_bytes: bytes) -> FaceAnalysis:
        """Decode, detect once, embed + score fake. Raises NoFaceError / ValueError(bad image)."""

    def verify(self, image_bytes: bytes, template: np.ndarray) -> VerifyResult:
        """Full decision against an enrolled template. Never raises NoFaceError (returns NO_FACE)."""


def make_template(embeddings: list[np.ndarray]) -> np.ndarray:
    """Mean of L2-normalised embeddings, re-normalised."""
    m = np.mean([e / np.linalg.norm(e) for e in embeddings], axis=0)
    return (m / np.linalg.norm(m)).astype(np.float32)
