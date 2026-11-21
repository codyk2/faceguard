"""Smoke tests with the real pretrained models (slow; run with `-m models` or the full suite)."""
from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import pytest

from app.config import settings
from app.ml.interfaces import NoFaceError
from app.ml.preprocess import MULTI_FACE_WARNING
from app.schemas import Decision

pytestmark = pytest.mark.models


def _jpg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def pipeline():
    from app.ml.pipeline import load_pipeline

    return load_pipeline()


@pytest.fixture(scope="module")
def sample_bytes():
    from insightface.data import get_image

    return _jpg(get_image("t1"))  # group photo shipped with insightface


def test_face_found_on_sample(pipeline, sample_bytes):
    fa = pipeline.analyze(sample_bytes)
    assert fa.embedding.shape == (512,) and fa.embedding.dtype == np.float32
    assert np.linalg.norm(fa.embedding) == pytest.approx(1.0, abs=1e-4)
    assert 0.0 <= fa.fake_probability <= 1.0
    assert fa.fake_label in pipeline.detector.id2label.values()
    assert fa.face_box.x2 > fa.face_box.x1 and fa.face_box.score > 0.5
    assert MULTI_FACE_WARNING in fa.warnings
    assert {"detect", "embed", "fake", "total"} <= set(fa.timings_ms)


def test_no_face_on_blank(pipeline):
    blank = _jpg(np.full((480, 640, 3), 255, np.uint8))
    with pytest.raises(NoFaceError):
        pipeline.analyze(blank)
    r = pipeline.verify(blank, np.ones(512, np.float32) / np.sqrt(512))
    assert r.decision is Decision.NO_FACE


def test_same_image_similarity_is_one(pipeline, sample_bytes):
    template = pipeline.analyze(sample_bytes).embedding
    r = pipeline.verify(sample_bytes, template)
    assert r.similarity == pytest.approx(1.0, abs=1e-4)
    assert r.decision in (Decision.ACCEPT, Decision.REJECT_FAKE)


def test_label_mapping_resolves(pipeline):
    det = pipeline.detector
    assert "fake" in det.id2label[det.fake_index].lower()
    assert "detector=" in pipeline.device and "verifier=cpu" in pipeline.device


def test_fallback_detector_label_mapping():
    from app.ml.detector import ViTFakeDetector

    det = ViTFakeDetector(settings.detector_fallback_id, device="cpu")
    assert "fake" in det.id2label[det.fake_index].lower()
