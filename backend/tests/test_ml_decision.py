"""Fast tests: decision table, maths and label mapping with stubbed components (no model loading)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.ml.detector import LabelMappingError, find_fake_index
from app.ml.interfaces import NoFaceError, make_template
from app.ml.pipeline import FaceGuardPipeline, decide
from app.ml.preprocess import DetectedFace, decode_image, expand_box
from app.ml.verifier import cosine_similarity, l2_normalize
from app.schemas import Decision, FaceBox

CFG = Settings(t_sim=0.35, t_fake=0.5)


def _png() -> bytes:
    ok, buf = cv2.imencode(".png", np.full((32, 32, 3), 127, np.uint8))
    assert ok
    return buf.tobytes()


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


class StubPre:
    def __init__(self, no_face=False, warnings=()):
        self.no_face, self.warnings = no_face, list(warnings)

    def detect(self, img):
        if self.no_face:
            raise NoFaceError("no face detected")
        return DetectedFace(
            face_box=FaceBox(x1=1, y1=2, x2=30, y2=31, score=0.9),
            kps=np.zeros((5, 2), np.float32),
            aligned_crop=np.zeros((112, 112, 3), np.uint8),
            fake_crop=Image.new("RGB", (40, 40)),
            warnings=list(self.warnings),
        )


class StubVerifier:
    name, device = "stub-arcface", "cpu"

    def __init__(self, emb):
        self.emb = emb

    def embed(self, crop):
        return self.emb


class StubDetector:
    name, device = "stub-vit", "mps"

    def __init__(self, p):
        self.p = p

    def predict(self, img):
        return self.p, "Fake" if self.p >= 0.5 else "Real"


def _pipe(emb, p_fake, **pre_kw):
    return FaceGuardPipeline(StubPre(**pre_kw), StubVerifier(emb), StubDetector(p_fake), CFG)


@pytest.mark.parametrize(
    "sim, p_fake, expected",
    [
        (0.90, 0.10, Decision.ACCEPT),
        (0.35, 0.49, Decision.ACCEPT),            # boundaries: sim >= t_sim, p < t_fake
        (0.90, 0.50, Decision.REJECT_FAKE),
        (0.10, 0.99, Decision.REJECT_FAKE),       # fake checked before mismatch
        (0.34, 0.10, Decision.REJECT_MISMATCH),
        (-0.2, 0.00, Decision.REJECT_MISMATCH),
    ],
)
def test_decide_table(sim, p_fake, expected):
    assert decide(sim, p_fake, CFG) is expected


def test_verify_accept_same_embedding():
    t = _unit(0)
    r = _pipe(t, 0.1).verify(_png(), t)
    assert r.decision is Decision.ACCEPT
    assert r.similarity == pytest.approx(1.0, abs=1e-5)
    assert r.fake_probability == pytest.approx(0.1)
    assert r.face_box is not None
    assert set(r.timings_ms) == {"detect", "embed", "fake", "total"}


def test_verify_mismatch_and_fake():
    assert _pipe(_unit(1), 0.1).verify(_png(), _unit(2)).decision is Decision.REJECT_MISMATCH
    assert _pipe(_unit(1), 0.9).verify(_png(), _unit(1)).decision is Decision.REJECT_FAKE


def test_verify_no_face_does_not_raise():
    r = _pipe(_unit(0), 0.1, no_face=True).verify(_png(), _unit(0))
    assert r.decision is Decision.NO_FACE
    assert r.similarity is None and r.fake_probability is None and r.face_box is None


def test_analyze_raises_no_face_and_passes_warnings():
    with pytest.raises(NoFaceError):
        _pipe(_unit(0), 0.1, no_face=True).analyze(_png())
    fa = _pipe(_unit(0), 0.2, warnings=["multiple faces detected; using the largest"]).analyze(_png())
    assert fa.warnings == ["multiple faces detected; using the largest"]
    assert fa.fake_label == "Real"


def test_bad_bytes_raise_value_error():
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    with pytest.raises(ValueError):
        _pipe(_unit(0), 0.1).verify(b"", _unit(0))


def test_pipeline_metadata():
    p = _pipe(_unit(0), 0.1)
    assert p.device == "verifier=cpu, detector=mps"
    assert p.verifier_name == "stub-arcface" and p.detector_name == "stub-vit"


def test_cosine_and_template_maths():
    a, b = _unit(3), _unit(4)
    assert cosine_similarity(a, a * 5) == pytest.approx(1.0, abs=1e-6)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0, abs=1e-6)
    assert np.linalg.norm(l2_normalize(a * 3)) == pytest.approx(1.0, abs=1e-6)
    t = make_template([a, b])
    assert t.dtype == np.float32 and np.linalg.norm(t) == pytest.approx(1.0, abs=1e-6)
    assert cosine_similarity(t, a) == pytest.approx(cosine_similarity(t, b), abs=1e-5)
    with pytest.raises(ValueError):
        l2_normalize(np.zeros(512))


@pytest.mark.parametrize(
    "id2label, idx",
    [
        ({0: "Realism", 1: "Deepfake"}, 1),
        ({0: "Fake", 1: "Real"}, 0),
        ({"0": "real", "1": "FAKE"}, 1),
    ],
)
def test_label_mapping(id2label, idx):
    assert find_fake_index(id2label) == idx


@pytest.mark.parametrize("id2label", [{0: "LABEL_0", 1: "LABEL_1"}, {0: "fake_a", 1: "deepfake_b"}])
def test_label_mapping_errors(id2label):
    with pytest.raises(LabelMappingError):
        find_fake_index(id2label)


def test_expand_box_clips_to_image():
    assert expand_box((10, 10, 110, 110), 0.3, 1000, 1000) == (0, 0, 125, 125)
    assert expand_box((100, 100, 200, 200), 0.3, 1000, 1000) == (85, 85, 215, 215)
