"""Request dependencies and the deterministic fake pipeline.

The fake lives here (not under tests/) so `FACEGUARD_FAKE_PIPELINE=1 uvicorn app.main:app`
works for frontend development without any models installed.

Fake pipeline steering (markers searched anywhere in the uploaded bytes):
  - not a recognised image (JPEG/PNG/GIF/WEBP/BMP magic) and no b"FG:" marker -> ValueError
  - b"FG:NOFACE"          -> NoFaceError (verify returns NO_FACE)
  - b"FG:FAKE"            -> fake_probability 0.9 (else 0.1)
  - b"FG:PERSON=<id>;"    -> embedding seeded by <id>; same id => similarity ~1, different => ~0
  - otherwise the embedding is seeded by sha256 of the whole file, so re-uploading the
    exact same real photo verifies as ACCEPT and any other photo as REJECT_MISMATCH.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
from fastapi import Request

from app.config import settings
from app.ml.interfaces import FaceAnalysis, NoFaceError, Pipeline, VerifyResult
from app.schemas import Decision, FaceBox
from app.store import Store

_IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"GIF87a", b"GIF89a", b"BM")
_PERSON_RE = re.compile(rb"FG:PERSON=([A-Za-z0-9_\-]+)")


def _looks_like_image(data: bytes) -> bool:
    if data.startswith(_IMAGE_MAGIC):
        return True
    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def _seeded_embedding(key: bytes) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


class FakePipeline:
    device = "cpu"
    verifier_name = "fake-verifier"
    detector_name = "fake-detector"

    def analyze(self, image_bytes: bytes) -> FaceAnalysis:
        if not image_bytes or not (_looks_like_image(image_bytes) or b"FG:" in image_bytes):
            raise ValueError("could not decode image")
        if b"FG:NOFACE" in image_bytes:
            raise NoFaceError("no face detected")
        m = _PERSON_RE.search(image_bytes)
        key = b"person:" + m.group(1) if m else b"file:" + image_bytes
        is_fake = b"FG:FAKE" in image_bytes
        return FaceAnalysis(
            embedding=_seeded_embedding(key),
            fake_probability=0.9 if is_fake else 0.1,
            fake_label="fake" if is_fake else "real",
            face_box=FaceBox(x1=10.0, y1=20.0, x2=110.0, y2=140.0, score=0.99),
            warnings=[],
            timings_ms={"detect": 1.0, "embed": 1.0, "fake": 1.0},
        )

    def verify(self, image_bytes: bytes, template: np.ndarray) -> VerifyResult:
        try:
            a = self.analyze(image_bytes)
        except NoFaceError:
            return VerifyResult(
                decision=Decision.NO_FACE, similarity=None, fake_probability=None,
                face_box=None, warnings=["no face detected"], timings_ms={"detect": 1.0},
            )
        sim = float(np.dot(a.embedding, template))
        if a.fake_probability >= settings.t_fake:
            decision = Decision.REJECT_FAKE
        elif sim < settings.t_sim:
            decision = Decision.REJECT_MISMATCH
        else:
            decision = Decision.ACCEPT
        return VerifyResult(
            decision=decision, similarity=sim, fake_probability=a.fake_probability,
            face_box=a.face_box, warnings=list(a.warnings), timings_ms=dict(a.timings_ms),
        )


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def get_store(request: Request) -> Store:
    return request.app.state.store
