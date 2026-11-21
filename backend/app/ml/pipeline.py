"""FaceGuard ML pipeline: one detection pass feeds both ArcFace and the ViT detector.

`load_pipeline()` is the only entry point the API layer should use. The class takes
its three components by injection so tests can pass lightweight stubs.
"""
from __future__ import annotations

import time

import numpy as np

from app.config import Settings, settings as default_settings
from app.ml.interfaces import FaceAnalysis, NoFaceError, VerifyResult
from app.schemas import Decision


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 2)


def decide(similarity: float, fake_probability: float, cfg: Settings) -> Decision:
    """Decision table (DESIGN.md section 2). REJECT_FAKE is checked first."""
    if fake_probability >= cfg.t_fake:
        return Decision.REJECT_FAKE
    if similarity < cfg.t_sim:
        return Decision.REJECT_MISMATCH
    return Decision.ACCEPT


class FaceGuardPipeline:
    def __init__(self, preprocessor, verifier, detector, cfg: Settings | None = None):
        self.cfg = cfg or default_settings
        self.preprocessor = preprocessor
        self.verifier = verifier
        self.detector = detector
        self.verifier_name: str = getattr(verifier, "name", type(verifier).__name__)
        self.detector_name: str = getattr(detector, "name", type(detector).__name__)
        self.device: str = (
            f"verifier={getattr(verifier, 'device', 'cpu')}, "
            f"detector={getattr(detector, 'device', 'cpu')}"
        )

    def analyze(self, image_bytes: bytes) -> FaceAnalysis:
        from app.ml.preprocess import decode_image

        t_total = time.perf_counter()
        t0 = time.perf_counter()
        img = decode_image(image_bytes)            # ValueError on bad bytes
        face = self.preprocessor.detect(img)       # NoFaceError if none / too small
        t_detect = _ms(t0)

        t0 = time.perf_counter()
        embedding = self.verifier.embed(face.aligned_crop)
        t_embed = _ms(t0)

        t0 = time.perf_counter()
        p_fake, label = self.detector.predict(face.fake_crop)
        t_fake = _ms(t0)

        return FaceAnalysis(
            embedding=embedding,
            fake_probability=float(p_fake),
            fake_label=label,
            face_box=face.face_box,
            warnings=list(face.warnings),
            timings_ms={"detect": t_detect, "embed": t_embed, "fake": t_fake,
                        "total": _ms(t_total)},
        )

    def verify(self, image_bytes: bytes, template: np.ndarray) -> VerifyResult:
        from app.ml.verifier import cosine_similarity

        t_total = time.perf_counter()
        try:
            fa = self.analyze(image_bytes)
        except NoFaceError as e:
            return VerifyResult(
                decision=Decision.NO_FACE, similarity=None, fake_probability=None,
                face_box=None, warnings=[str(e)] if str(e) else [],
                timings_ms={"total": _ms(t_total)},
            )
        similarity = cosine_similarity(fa.embedding, template)
        timings = dict(fa.timings_ms)
        timings["total"] = _ms(t_total)
        return VerifyResult(
            decision=decide(similarity, fa.fake_probability, self.cfg),
            similarity=similarity,
            fake_probability=fa.fake_probability,
            face_box=fa.face_box,
            warnings=fa.warnings,
            timings_ms=timings,
        )


def load_pipeline(detector_model_id: str | None = None, cfg: Settings | None = None) -> FaceGuardPipeline:
    """Load InsightFace (CPU, ONNX Runtime) and the ViT detector (MPS if available)."""
    from app.ml.detector import ViTFakeDetector
    from app.ml.preprocess import FacePreprocessor
    from app.ml.verifier import ArcFaceVerifier

    cfg = cfg or default_settings
    pre = FacePreprocessor(cfg)
    verifier = ArcFaceVerifier(pre.rec_model, device=pre.device)
    detector = ViTFakeDetector(detector_model_id or cfg.detector_model_id)
    return FaceGuardPipeline(pre, verifier, detector, cfg)
