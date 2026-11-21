"""Milestone 1 inference test: load both models and print a screenshot-friendly table.

Usage (from backend/):
    uv run python ../scripts/inference_test.py [--detector <hf model id>]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore")

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import DATA_DIR, settings  # noqa: E402
from app.ml.interfaces import NoFaceError  # noqa: E402
from app.ml.verifier import cosine_similarity  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
W = 78


def rule(ch: str = "-") -> None:
    print(ch * W)


def short(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "~"


def jpg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


def images_in(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.is_dir() else []


# ---------------------------------------------------------------- verification pairs
def lfw_pairs() -> list[tuple[str, str, bytes, bytes, bool]] | None:
    root = DATA_DIR / "lfw"
    if not root.is_dir():
        return None
    people: dict[str, list[Path]] = {}
    for d in root.rglob("*"):
        if d.is_dir():
            imgs = images_in(d)
            if imgs:
                people[d.name] = imgs
    multi = sorted((n for n, v in people.items() if len(v) >= 2), key=lambda n: (-len(people[n]), n))
    if len(multi) < 4:
        return None
    a, b, c, d = multi[:4]
    rd = lambda p: p.read_bytes()  # noqa: E731
    return [
        (f"{a}/1", f"{a}/2", rd(people[a][0]), rd(people[a][1]), True),
        (f"{b}/1", f"{b}/2", rd(people[b][0]), rd(people[b][1]), True),
        (f"{a}/1", f"{c}/1", rd(people[a][0]), rd(people[c][0]), False),
        (f"{b}/1", f"{d}/1", rd(people[b][0]), rd(people[d][0]), False),
    ]


def sample_pairs(pipe) -> list[tuple[str, str, bytes, bytes, bool]]:
    """Fallback built from insightface's bundled images (t1 group photo, Tom Hanks)."""
    from insightface.data import get_image
    from app.ml.preprocess import expand_box

    hanks = get_image("Tom_Hanks_54745")
    t1 = get_image("t1")
    boxes, _ = pipe.preprocessor.det_model.detect(t1, max_num=0, metric="default")
    boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:2]
    h, w = t1.shape[:2]
    crops = []
    for b in boxes:
        x1, y1, x2, y2 = expand_box(tuple(b[:4]), 1.0, w, h)
        crops.append(cv2.resize(t1[y1:y2, x1:x2], None, fx=2.0, fy=2.0))
    aug = lambda im: cv2.convertScaleAbs(cv2.flip(im, 1), alpha=1.1, beta=10)  # noqa: E731
    return [
        ("tom_hanks", "tom_hanks(mirror)", jpg(hanks), jpg(aug(hanks)), True),
        ("t1_face0", "t1_face0(mirror)", jpg(crops[0]), jpg(aug(crops[0])), True),
        ("tom_hanks", "t1_face0", jpg(hanks), jpg(crops[0]), False),
        ("t1_face0", "t1_face1", jpg(crops[0]), jpg(crops[1]), False),
    ]


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--detector", default=None, help="HF model id for the fake detector")
    ap.add_argument("-n", type=int, default=5, help="images per class for detection")
    args = ap.parse_args()

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):  # library chatter
        from transformers.utils import logging as hf_logging

        from app.ml.detector import ViTFakeDetector
        from app.ml.pipeline import FaceGuardPipeline
        from app.ml.preprocess import FacePreprocessor, decode_image
        from app.ml.verifier import ArcFaceVerifier

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        t0 = time.perf_counter()
        pre = FacePreprocessor(settings)
        verifier = ArcFaceVerifier(pre.rec_model, device=pre.device)
        t_ver = time.perf_counter() - t0
        t0 = time.perf_counter()
        detector = ViTFakeDetector(args.detector or settings.detector_model_id)
        t_det = time.perf_counter() - t0
    pipe = FaceGuardPipeline(pre, verifier, detector, settings)

    rule("=")
    print("FaceGuard inference test".center(W))
    rule("=")
    print(f"Verifier : {pipe.verifier_name}")
    print(f"           device=cpu (ONNX Runtime), load {t_ver:.1f}s")
    print(f"Detector : {detector.model_id}")
    print(f"           device={detector.device} (PyTorch), load {t_det:.1f}s")
    print(f"           id2label={detector.id2label}  fake_index={detector.fake_index}")
    print(f"Thresholds: T_SIM={settings.t_sim}  T_FAKE={settings.t_fake}")

    # (a) verification
    pairs = lfw_pairs()
    src = "LFW" if pairs else "insightface samples (data/lfw missing)"
    if pairs is None:
        pairs = sample_pairs(pipe)
    rule()
    print(f"(a) Verification - ArcFace cosine similarity [{src}]")
    rule()
    print(f"{'Image A':<24} {'Image B':<24} {'Truth':<6} {'Cosine':>7}  {'Result':<9} ")
    for na, nb, ba, bb, same in pairs:
        try:
            ea = pipe.analyze(ba).embedding
            eb = pipe.analyze(bb).embedding
            sim = cosine_similarity(ea, eb)
            res = "MATCH" if sim >= settings.t_sim else "no-match"
            ok = "ok" if (sim >= settings.t_sim) == same else "WRONG"
            print(f"{short(na, 24):<24} {short(nb, 24):<24} {'same' if same else 'diff':<6} "
                  f"{sim:>7.3f}  {res:<9} {ok}")
        except NoFaceError as e:
            print(f"{short(na, 24):<24} {short(nb, 24):<24} {'same' if same else 'diff':<6} "
                  f"{'-':>7}  NO_FACE   ({e})")

    # (b) detection
    rule()
    print(f"(b) Fake detection - P(fake) from {short(detector.model_id, 50)}")
    rule()
    fakes = DATA_DIR / "fakes"
    real_imgs = images_in(fakes / "real")[: args.n]
    # Round-robin across fake generation methods (filename prefix: inpainting/insight/text2img)
    by_kind: dict[str, list] = {}
    for p in images_in(fakes / "fake"):
        by_kind.setdefault(p.name.split("_")[0], []).append(p)
    fake_imgs, i = [], 0
    while len(fake_imgs) < args.n and any(i < len(v) for v in by_kind.values()):
        fake_imgs += [v[i] for v in by_kind.values() if i < len(v)][: args.n - len(fake_imgs)]
        i += 1
    if not real_imgs and not fake_imgs:
        print("NOTE: data/fakes/real and data/fakes/fake not found - detection on")
        print("      insightface samples only (all real photos, no fake samples).")
        from insightface.data import get_image
        rows = [("real", "t1.jpg", jpg(get_image("t1"))),
                ("real", "Tom_Hanks_54745.png", jpg(get_image("Tom_Hanks_54745")))]
    else:
        rows = [("real", p.name, p.read_bytes()) for p in real_imgs]
        rows += [("fake", p.name, p.read_bytes()) for p in fake_imgs]
    print(f"{'Truth':<6} {'File':<34} {'P(fake)':>8}  {'Predicted':<12} {'Crop':<5}")
    per_class: dict[str, list[float]] = {"real": [], "fake": []}
    for truth, name, data in rows:
        crop = "face"
        try:
            fa = pipe.analyze(data)
            p, label = fa.fake_probability, fa.fake_label
        except NoFaceError:
            from PIL import Image
            img = cv2.cvtColor(decode_image(data), cv2.COLOR_BGR2RGB)
            p, label = detector.predict(Image.fromarray(img))
            crop = "full"
        per_class[truth].append(p)
        flag = "ok" if (p >= settings.t_fake) == (truth == "fake") else "WRONG"
        print(f"{truth:<6} {short(name, 34):<34} {p:>8.3f}  {short(label, 12):<12} {crop:<5} {flag}")
    rule()
    means = "  ".join(f"mean P(fake|{k})={np.mean(v):.3f}" for k, v in per_class.items() if v)
    print(means)
    rule("=")


if __name__ == "__main__":
    main()
