"""Fine-tune the ViT fake detector on DeepFakeFace (DFF), then evaluate it.

Three steps (run from backend/):

  uv run python ../scripts/train_detector.py crops   # face crops via the app's own preprocessing
  uv run python ../scripts/train_detector.py train   # fine-tune, pick best epoch on validation
  uv run python ../scripts/train_detector.py eval    # held-out DFF test set + LFW real photos

Data splits (no leakage):
  train / val : data/fakes_train (paired real+fake per source photo), split 90/10 by source id
  test        : data/fakes       (the 600-image evaluation set; source ids disjoint from train)
  lfw-real    : 1,000 random LFW photos (all real) - out-of-domain false-positive check

Crops come from FacePreprocessor.detect(...).fake_crop, i.e. exactly what the app feeds the
ViT at inference time. Images whose face is missing or < min_face_px are skipped, as in the app.

Output model: data/models/detector-dff-ft/ (save_pretrained format; loadable by
ViTFakeDetector(model_id=<that path>)). Metrics: data/eval/detector_ft.json and .md.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import _common as C  # sets sys.path for `app` and NO_ALBUMENTATIONS_UPDATE

import numpy as np
from PIL import Image

CROPS = C.DATA_DIR / "crops"
TRAIN_DIR = C.DATA_DIR / "fakes_train"
MODELS_DIR = C.DATA_DIR / "models"
OUT_DIR = MODELS_DIR / "detector-dff-ft"  # overridden by --out
# resolution-loss augmentation; "strong" (--degrade strong) targets LFW-like face sizes
DEGRADE = {"mode": "mild", "p": 0.3, "min_px": 72, "max_px": 200}
BASE_MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"
SEED = 2026


# --------------------------------------------------------------------------- crops
def _kind(name: str) -> str:
    return name.split("_", 1)[0]  # wiki / inpainting / insight / text2img


def cmd_crops(args) -> None:
    import cv2
    from app.ml.interfaces import NoFaceError
    from app.ml.preprocess import FacePreprocessor

    pre = FacePreprocessor()
    jobs: list[tuple[str, Path, str, str]] = []  # split, src, label, group

    with open(TRAIN_DIR / "manifest.csv") as f:
        for r in csv.DictReader(f):
            jobs.append(("trainval", TRAIN_DIR / r["file"], r["label"], r["source_id"]))
    with open(C.FAKES_DIR / "manifest.csv") as f:
        for r in csv.DictReader(f):
            jobs.append(("test", C.FAKES_DIR / r["file"], r["label"], r["member"].split("/", 1)[1]))
    lfw = sorted(C.LFW_DIR.glob("*/*.jpg"))
    random.Random(SEED).shuffle(lfw)
    for p in lfw[: args.n_lfw]:
        jobs.append(("lfw", p, "real", p.parent.name))

    rows, skipped, t0 = [], 0, time.time()
    for n, (split, src, label, group) in enumerate(jobs, 1):
        dest = CROPS / split / label / (src.stem + ".png")
        if not dest.exists():
            img = cv2.imread(str(src))
            if img is None:
                skipped += 1
                continue
            try:
                face = pre.detect(img)
            except NoFaceError:
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            face.fake_crop.save(dest)  # PNG: lossless, so no extra JPEG artifacts per class
        rows.append({"split": split, "path": str(dest.relative_to(CROPS)), "label": label,
                     "kind": "lfw" if split == "lfw" else _kind(src.name), "group": group})
        if n % 500 == 0:
            print(f"  {n}/{len(jobs)} ({time.time() - t0:.0f}s, skipped {skipped})", flush=True)

    with open(CROPS / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        k = f"{r['split']}/{r['label']}"
        counts[k] = counts.get(k, 0) + 1
    print(f"crops: {counts}; skipped (no usable face) {skipped}")


def load_index() -> list[dict]:
    with open(CROPS / "index.csv") as f:
        return list(csv.DictReader(f))


def split_trainval(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    tv = [r for r in rows if r["split"] == "trainval"]
    groups = sorted({r["group"] for r in tv})
    random.Random(SEED).shuffle(groups)
    val_groups = set(groups[: len(groups) // 10])
    return ([r for r in tv if r["group"] not in val_groups],
            [r for r in tv if r["group"] in val_groups])


# --------------------------------------------------------------------------- data
class CropDataset:
    """Face crops -> normalised tensors. Augmentations are applied to BOTH classes so
    compression / resolution can't separate them; only generation artifacts can."""

    def __init__(self, rows, mean, std, size=224, train=False):
        self.rows, self.mean, self.std, self.size, self.train = rows, mean, std, size, train

    def __len__(self):
        return len(self.rows)

    def _augment(self, im: Image.Image, rng: random.Random) -> Image.Image:
        w, h = im.size
        s = rng.uniform(0.80, 1.0)  # random crop of 80-100% of the side
        cw, ch = int(w * s), int(h * s)
        x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
        im = im.crop((x, y, x + cw, y + ch))
        if rng.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        # Aspect jitter on BOTH classes. DFF inpainting / face-swap fakes were stretched to
        # 512x512 (faces ~12-30% wider than in the source photo), so face shape alone would
        # separate the classes. Random stretch/squeeze makes shape uninformative.
        a = rng.uniform(0.75, 1.35)
        im = im.resize((max(16, int(im.width * a)), im.height), Image.BILINEAR)
        if rng.random() < DEGRADE["p"]:  # resolution loss, applied to both classes
            if DEGRADE["mode"] == "strong":
                # shrink the crop's long side to an LFW-like size (small, soft faces), so
                # low resolution alone is not evidence of real or fake
                target = rng.randint(DEGRADE["min_px"], DEGRADE["max_px"])
                f = min(1.0, target / max(im.size))
            else:
                f = rng.uniform(0.4, 0.9)
            im = im.resize((max(16, int(im.width * f)), max(16, int(im.height * f))),
                           rng.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]))
        if rng.random() < 0.5:  # re-compression
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=rng.randint(55, 95))
            im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        return im

    def __getitem__(self, i):
        import torch

        r = self.rows[i]
        im = Image.open(CROPS / r["path"]).convert("RGB")
        if self.train:
            im = self._augment(im, random.Random())
        im = im.resize((self.size, self.size), Image.BICUBIC)
        x = (np.asarray(im, dtype=np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(x.transpose(2, 0, 1)), int(r["label"] == "fake")


def make_loader(ds, batch, shuffle):
    import torch

    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)


def predict(model, loader, device, fake_index):
    import torch

    model.eval()
    ps, ys = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(pixel_values=x.to(device)).logits.float()
            ps.append(torch.softmax(logits, -1)[:, fake_index].cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def auc(p, y) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan")


# --------------------------------------------------------------------------- train
def cmd_train(args) -> None:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from app.ml.detector import find_fake_index, pick_device

    torch.manual_seed(SEED)
    random.seed(SEED)
    device = pick_device()
    proc = AutoImageProcessor.from_pretrained(BASE_MODEL)
    model = AutoModelForImageClassification.from_pretrained(BASE_MODEL).to(device)
    fake_index = find_fake_index(model.config.id2label)
    mean, std = np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)

    rows = load_index()
    tr, va = split_trainval(rows)
    # remap labels to the model's own index order
    if fake_index != 1:
        raise SystemExit(f"unexpected fake_index {fake_index}; adjust label mapping")
    tl = make_loader(CropDataset(tr, mean, std, train=True), args.batch, True)
    vl = make_loader(CropDataset(va, mean, std), 64, False)
    print(f"device={device} train={len(tr)} val={len(va)} epochs={args.epochs} batch={args.batch}")

    p0, y0 = predict(model, vl, device, fake_index)
    print(f"epoch 0 (base model) val AUC {auc(p0, y0):.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * len(tl)
    warm = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    best, history = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for step, (x, y) in enumerate(tl, 1):
            logits = model(pixel_values=x.to(device)).logits
            loss = lossf(logits.float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            if step % 20 == 0:
                print(f"  ep{ep} step {step}/{len(tl)} loss {run / step:.4f} ({time.time() - t0:.0f}s)", flush=True)
        p, y = predict(model, vl, device, fake_index)
        a = auc(p, y)
        acc = float(((p >= 0.5) == y).mean())
        history.append({"epoch": ep, "train_loss": run / len(tl), "val_auc": a, "val_acc": acc})
        print(f"epoch {ep}: train loss {run / len(tl):.4f}  val AUC {a:.4f}  val acc {acc:.4f}  ({time.time() - t0:.0f}s)")
        if a > best:
            best = a
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUT_DIR)
            proc.save_pretrained(OUT_DIR)
            print(f"  saved best -> {OUT_DIR}")
    (OUT_DIR / "train_history.json").write_text(json.dumps(
        {"base_model": BASE_MODEL, "train": len(tr), "val": len(va), "args": vars(args) | {"func": None},
         "history": history, "best_val_auc": best}, indent=2, default=str))


# --------------------------------------------------------------------------- eval
def _metrics(p, y, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return {"acc": (tp + tn) / len(y), "tpr": tp / max(1, tp + fn), "fpr": fp / max(1, fp + tn),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def cmd_eval(args) -> None:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from app.ml.detector import find_fake_index, pick_device

    rows = load_index()
    _, va = split_trainval(rows)
    test = [r for r in rows if r["split"] == "test"]
    lfw = [r for r in rows if r["split"] == "lfw"]
    device = pick_device()
    out = {}
    candidates = [("base", BASE_MODEL)] + [
        (d.name, str(d)) for d in sorted(MODELS_DIR.glob("detector-*")) if (d / "config.json").exists()]
    for name, mid in candidates:
        proc = AutoImageProcessor.from_pretrained(mid)
        model = AutoModelForImageClassification.from_pretrained(mid).to(device)
        fi = find_fake_index(model.config.id2label)
        mean, std = np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)
        pv, yv = predict(model, make_loader(CropDataset(va, mean, std), 64, False), device, fi)
        pt, yt = predict(model, make_loader(CropDataset(test, mean, std), 64, False), device, fi)
        pl, _ = predict(model, make_loader(CropDataset(lfw, mean, std), 64, False), device, fi)
        # threshold chosen on VALIDATION reals only: 5% false-positive rate
        t5 = float(np.quantile(pv[yv == 0], 0.95))
        res = {"val_auc": auc(pv, yv), "test_auc": auc(pt, yt), "t_fpr5_val": t5}
        for tname, t in (("t0.5", 0.5), ("t_fpr5", t5)):
            m = _metrics(pt, yt, t)
            kinds = {}
            for k in ("inpainting", "insight", "text2img"):
                idx = np.array([r["kind"] == k for r in test])
                kinds[k] = float((pt[idx] >= t).mean())
            res[tname] = m | {"caught_by_kind": kinds,
                              "lfw_real_flagged": float((pl >= t).mean()), "threshold": t}
        out[name] = res
        print(f"[{name}] val AUC {res['val_auc']:.4f}  test AUC {res['test_auc']:.4f}  "
              f"t_fpr5={t5:.3f}")
        for tname in ("t0.5", "t_fpr5"):
            m = res[tname]
            print(f"   @{tname}: test acc {m['acc']:.3f}  TPR {m['tpr']:.3f}  FPR {m['fpr']:.3f}  "
                  f"LFW real flagged {m['lfw_real_flagged']:.3f}  by kind {m['caught_by_kind']}")
    # Shortcut baseline: how well does crop aspect ratio ALONE separate real from fake?
    asp = np.array([Image.open(CROPS / r["path"]).width / Image.open(CROPS / r["path"]).height for r in test])
    yt_ = np.array([int(r["label"] == "fake") for r in test])
    out["aspect_only_test_auc"] = max(auc(asp, yt_), 1 - auc(asp, yt_))
    print(f"[shortcut baseline] crop aspect ratio alone: test AUC {out['aspect_only_test_auc']:.4f}")
    out["counts"] = {"val": len(va), "test": len(test), "lfw": len(lfw)}
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (C.EVAL_DIR / "detector_ft.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {C.EVAL_DIR / 'detector_ft.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops"); c.add_argument("--n-lfw", type=int, default=1000); c.set_defaults(func=cmd_crops)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--out", default="detector-dff-ft", help="folder name under data/models/")
    t.add_argument("--degrade", choices=["mild", "strong"], default="mild")
    t.set_defaults(func=cmd_train)
    e = sub.add_parser("eval"); e.set_defaults(func=cmd_eval)
    args = ap.parse_args()
    if getattr(args, "out", None):
        global OUT_DIR
        OUT_DIR = MODELS_DIR / args.out
    if getattr(args, "degrade", None) == "strong":
        DEGRADE.update(mode="strong", p=0.6)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
