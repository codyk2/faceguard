"""Offline evaluation of FaceGuard: verification on LFW pairs, fake detection on data/fakes.

Runs directly on the ML pipeline (no HTTP). Usage (from backend/):
    uv run python ../scripts/evaluate.py                      # full run, configured detector
    uv run python ../scripts/evaluate.py --limit 100 --compare
    uv run python ../scripts/evaluate.py --detector dima806/deepfake_vs_real_image_detection
    uv run python ../scripts/evaluate.py --skip-detection     # or --skip-verification

--limit N   N genuine + N impostor pairs, and N real + N fake images (fixed seed).
--compare   evaluate both candidate detectors (config detector_model_id and
            detector_fallback_id) on the same face crops.

Outputs in data/eval/: summary.json, summary.md, roc_verification.png,
roc_detection.png, confusion_<model>.png, scores_<model>.png, and caches
(embeddings_<pack>.npz, detector_<model>.json) so reruns only compute what is new.
Caches are keyed by image path only: pass --fresh after changing app/ml code.

Images where no usable face is found are counted as failures to acquire:
in verification such a pair gets similarity -1 (always rejected), in detection
the image is excluded from the metrics and the count is reported.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import _common as C
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from app.config import settings  # noqa: E402
from app.ml.interfaces import NoFaceError  # noqa: E402

EVAL = C.EVAL_DIR


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")


def short(model_id: str) -> str:
    return model_id.split("/")[0]


# ---------------------------------------------------------------- model loading
class Models:
    """Lazy holder: InsightFace and ViTs load only if a cache miss needs them."""

    def __init__(self, default_detector: str):
        self.default_detector = default_detector
        self._pipe = None
        self._detectors: dict[str, object] = {}

    @property
    def pipe(self):
        if self._pipe is None:
            from app.ml.pipeline import load_pipeline

            t0 = time.time()
            print(f"  loading pipeline (detector={self.default_detector}) ...", flush=True)
            self._pipe = load_pipeline(detector_model_id=self.default_detector)
            self._detectors[self.default_detector] = self._pipe.detector
            print(f"  loaded in {time.time() - t0:.1f}s; device: {self._pipe.device}", flush=True)
        return self._pipe

    def detector(self, model_id: str):
        if model_id not in self._detectors:
            if model_id == self.default_detector:
                _ = self.pipe
            else:
                from app.ml.detector import ViTFakeDetector

                print(f"  loading detector {model_id} ...", flush=True)
                self._detectors[model_id] = ViTFakeDetector(model_id)
        return self._detectors[model_id]

    def detect_face(self, path: Path):
        from app.ml.preprocess import decode_image

        return self.pipe.preprocessor.detect(decode_image(path.read_bytes()))


# ---------------------------------------------------------------- verification
def load_pairs(limit: int | None, seed: int, pairs_file: str | None) -> tuple[list, str]:
    """[(path_a, path_b, is_genuine)], description."""
    candidates = [pairs_file] if pairs_file else ["pairs.txt", "pairsDevTest.txt"]
    for name in candidates:
        p = C.LFW_DIR / name
        if not p.exists():
            continue
        lines = [ln.split() for ln in p.read_text().splitlines()]
        gen, imp = [], []
        for t in lines[1:]:
            if len(t) == 3:
                gen.append((C.lfw_image_path(t[0], int(t[1])), C.lfw_image_path(t[0], int(t[2])), 1))
            elif len(t) == 4:
                imp.append((C.lfw_image_path(t[0], int(t[1])), C.lfw_image_path(t[2], int(t[3])), 0))
        desc = f"LFW {name} ({len(gen)} genuine + {len(imp)} impostor)"
        if limit and limit < max(len(gen), len(imp)):
            rng = random.Random(seed)
            gen, imp = rng.sample(gen, min(limit, len(gen))), rng.sample(imp, min(limit, len(imp)))
            desc += f", subsampled to {len(gen)} + {len(imp)} (seed {seed})"
        return gen + imp, desc

    # no pairs file: sample 300 + 300 with a fixed seed
    people = C.lfw_people()
    if not people:
        raise SystemExit(f"no LFW data under {C.LFW_DIR}; run download_data.py --only lfw")
    n = limit or 300
    rng = random.Random(seed)
    multi = [k for k, v in people.items() if len(v) >= 2]
    names = list(people)
    gen, imp = [], []
    while len(gen) < n:
        a, b = rng.sample(people[rng.choice(multi)], 2)
        gen.append((a, b, 1))
    while len(imp) < n:
        x, y = rng.sample(names, 2)
        imp.append((rng.choice(people[x]), rng.choice(people[y]), 0))
    return gen + imp, f"random LFW pairs ({n} genuine + {n} impostor, seed {seed})"


def embed_all(paths: list[Path], models: Models) -> dict[str, np.ndarray | None]:
    cache_path = EVAL / f"embeddings_{settings.insightface_pack}.npz"
    cache: dict[str, np.ndarray | None] = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        for key, e in zip(z["keys"], z["embs"]):
            cache[str(key)] = None if np.isnan(e).any() else e
    todo = [p for p in dict.fromkeys(paths) if rel(p) not in cache]
    print(f"  embeddings: {len(cache)} cached, {len(todo)} to compute", flush=True)

    def save():
        keys = np.array(list(cache.keys()))
        embs = np.stack([v if v is not None else np.full(512, np.nan, np.float32)
                         for v in cache.values()]) if cache else np.zeros((0, 512), np.float32)
        np.savez(cache_path, keys=keys, embs=embs)

    t0 = time.time()
    for i, p in enumerate(todo, 1):
        try:
            face = models.detect_face(p)
            cache[rel(p)] = models.pipe.verifier.embed(face.aligned_crop)
        except (NoFaceError, ValueError):
            cache[rel(p)] = None
        if i % 200 == 0 or i == len(todo):
            save()
            print(f"    {i}/{len(todo)} ({(time.time() - t0) / i * 1000:.0f} ms/img)", flush=True)
    return cache


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(C.DATA_DIR.resolve()))


def verification(args, models: Models) -> dict:
    print("[verification]", flush=True)
    pairs, desc = load_pairs(args.limit, args.seed, args.pairs_file)
    missing = [p for a, b, _ in pairs for p in (a, b) if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} pair images missing, e.g. {missing[0]}")
    embs = embed_all([p for a, b, _ in pairs for p in (a, b)], models)
    y, s, fta = [], [], 0
    for a, b, g in pairs:
        ea, eb = embs[rel(a)], embs[rel(b)]
        if ea is None or eb is None:
            fta += 1
            sim = -1.0
        else:
            sim = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))
        y.append(g)
        s.append(sim)
    y, s = np.array(y), np.array(s)
    gen, imp = s[y == 1], s[y == 0]

    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    # EER: sweep every observed score as threshold; at the point where |FAR - FRR| is
    # smallest (ties broken by the lower mean), EER = (FAR + FRR) / 2
    cand = np.unique(s)
    far_c = (imp[None, :] >= cand[:, None]).mean(1)
    frr_c = (gen[None, :] < cand[:, None]).mean(1)
    order = np.lexsort(((far_c + frr_c), np.abs(far_c - frr_c)))
    i_eer = int(order[0])
    eer = float((far_c[i_eer] + frr_c[i_eer]) / 2)
    eer_thr = float(cand[i_eer])
    t = settings.t_sim
    far_t = float(np.mean(imp >= t))
    frr_t = float(np.mean(gen < t))
    # threshold for FAR <= 1%: the lowest t at which at most 1% of impostors score >= t,
    # i.e. just above the (k+1)-th highest impostor score, k = floor(1% * n_impostor)
    imp_desc = np.sort(imp)[::-1]
    k = int(np.floor(0.01 * len(imp_desc)))
    t_far1 = float(np.nextafter(imp_desc[k], np.inf)) if k < len(imp_desc) else float(imp_desc[-1])
    acc = (y == (s >= t)).mean()
    res = {
        "pairs": desc,
        "n_genuine": int((y == 1).sum()), "n_impostor": int((y == 0).sum()),
        "failure_to_acquire_pairs": fta,
        "roc_auc": float(roc_auc_score(y, s)),
        "eer": eer, "eer_threshold": eer_thr,
        "t_sim": t, "far_at_t_sim": far_t, "frr_at_t_sim": frr_t, "accuracy_at_t_sim": float(acc),
        "t_sim_for_far_1pct": t_far1,
        "far_at_t_far1": float(np.mean(imp >= t_far1)),
        "frr_at_t_far1": float(np.mean(gen < t_far1)),
        "genuine_sim_mean": float(gen.mean()), "impostor_sim_mean": float(imp.mean()),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(fpr, tpr, lw=2, color="#2563eb", label=f"ArcFace (AUC {res['roc_auc']:.4f})")
    ax1.plot([0, 1], [0, 1], ls="--", color="#9ca3af", lw=1)
    ax1.scatter([far_t], [1 - frr_t], color="#dc2626", zorder=5, label=f"t_sim={t:.2f}")
    ax1.scatter([res["far_at_t_far1"]], [1 - res["frr_at_t_far1"]], color="#16a34a", zorder=5,
                label=f"FAR=1% (t={t_far1:.3f})")
    ax1.set_xscale("log")
    ax1.set_xlim(1e-3, 1)
    ax1.set_xlabel("False accept rate (log)")
    ax1.set_ylabel("True accept rate")
    ax1.set_title(f"Verification ROC, EER {eer * 100:.2f}%")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)
    bins = np.linspace(-0.3, 1.0, 60)
    ax2.hist(imp, bins=bins, alpha=0.6, label="impostor", color="#f97316")
    ax2.hist(gen, bins=bins, alpha=0.6, label="genuine", color="#2563eb")
    ax2.axvline(t, color="#dc2626", ls="--", label=f"t_sim={t:.2f}")
    ax2.axvline(t_far1, color="#16a34a", ls=":", label=f"FAR=1% t={t_far1:.3f}")
    ax2.set_xlabel("Cosine similarity")
    ax2.set_ylabel("Pairs")
    ax2.set_title("Score distributions")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(EVAL / "roc_verification.png", dpi=150)
    plt.close(fig)
    res["plot"] = "data/eval/roc_verification.png"
    return res


# ---------------------------------------------------------------- detection
def load_fake_set(limit: int | None, seed: int) -> list[tuple[Path, int, str]]:
    """[(path, is_fake, source)] from data/fakes/{real,fake}."""
    src = {}
    man = C.FAKES_DIR / "manifest.csv"
    if man.exists():
        with open(man) as f:
            src = {r["file"]: r["source_zip"].rsplit(".", 1)[0] for r in csv.DictReader(f)}
    items = []
    for lab, y in (("real", 0), ("fake", 1)):
        files = sorted((C.FAKES_DIR / lab).glob("*.jpg"))
        if limit and limit < len(files):
            files = sorted(random.Random(seed).sample(files, limit))
        items += [(p, y, src.get(f"{lab}/{p.name}", lab)) for p in files]
    if not items:
        raise SystemExit(f"no images under {C.FAKES_DIR}; run download_data.py --only fakes")
    return items


def score_detector(model_id: str, items, models: Models) -> dict[str, float | None]:
    cache_path = EVAL / f"detector_{slug(model_id)}.json"
    cache: dict[str, float | None] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [p for p, _, _ in items if rel(p) not in cache]
    print(f"  {model_id}: {len(items) - len(todo)} cached, {len(todo)} to score", flush=True)
    if todo:
        det = models.detector(model_id)
        t0 = time.time()
        for i, p in enumerate(todo, 1):
            try:
                face = models.detect_face(p)
                cache[rel(p)], _ = det.predict(face.fake_crop)
            except (NoFaceError, ValueError):
                cache[rel(p)] = None
            if i % 100 == 0 or i == len(todo):
                cache_path.write_text(json.dumps(cache))
                print(f"    {i}/{len(todo)} ({(time.time() - t0) / i * 1000:.0f} ms/img)", flush=True)
    return cache


def detection(args, models: Models, model_ids: list[str]) -> dict:
    print("[detection]", flush=True)
    items = load_fake_set(args.limit, args.seed)
    t = settings.t_fake
    out = {"t_fake": t, "n_real": sum(1 for _, y, _ in items if y == 0),
           "n_fake": sum(1 for _, y, _ in items if y == 1), "models": {}}
    roc_fig, roc_ax = plt.subplots(figsize=(5.5, 5))
    colors = ["#2563eb", "#f97316", "#16a34a", "#9333ea"]
    for ci, mid in enumerate(model_ids):
        scores = score_detector(mid, items, models)
        kept = [(scores[rel(p)], y, s) for p, y, s in items if scores.get(rel(p)) is not None]
        p_fake = np.array([k[0] for k in kept])
        y = np.array([k[1] for k in kept])
        srcs = [k[2] for k in kept]
        pred = (p_fake >= t).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fpr, tpr, thr = roc_curve(y, p_fake)
        j = int(np.argmax(tpr - fpr))
        per_src = {}
        for s_name in sorted(set(srcs)):
            m = np.array([s == s_name for s in srcs])
            per_src[s_name] = {"n": int(m.sum()),
                               "flagged_fake_rate": float(pred[m].mean()),
                               "mean_p_fake": float(p_fake[m].mean())}
        r = {
            "no_face": len(items) - len(kept),
            "n_scored": len(kept),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p_fake)) if len(set(y)) == 2 else None,
            "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                                 "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
            "best_youden_threshold": float(thr[j]),
            "accuracy_at_best_threshold": float(accuracy_score(y, (p_fake >= thr[j]).astype(int))),
            "per_source": per_src,
        }
        out["models"][mid] = r

        roc_ax.plot(fpr, tpr, lw=2, color=colors[ci % 4], label=f"{short(mid)} (AUC {r['roc_auc']:.3f})")
        # confusion matrix + score histogram for this model
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.imshow(cm, cmap="Blues")
        for (i, k), v in np.ndenumerate(cm):
            a1.text(k, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=14)
        a1.set_xticks([0, 1], ["pred real", "pred fake"])
        a1.set_yticks([0, 1], ["real", "fake"])
        a1.set_title(f"{short(mid)} @ t_fake={t:.2f}\nacc {r['accuracy']:.3f}")
        bins = np.linspace(0, 1, 41)
        a2.hist(p_fake[y == 0], bins=bins, alpha=0.6, color="#2563eb", label="real")
        a2.hist(p_fake[y == 1], bins=bins, alpha=0.6, color="#f97316", label="fake")
        a2.axvline(t, color="#dc2626", ls="--", label=f"t_fake={t:.2f}")
        a2.set_xlabel("P(fake)")
        a2.set_ylabel("Images")
        a2.set_title("P(fake) distribution")
        a2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(EVAL / f"detection_{slug(mid)}.png", dpi=150)
        plt.close(fig)
        r["plot"] = f"data/eval/detection_{slug(mid)}.png"

    roc_ax.plot([0, 1], [0, 1], ls="--", color="#9ca3af", lw=1)
    roc_ax.set_xlabel("False positive rate (real flagged fake)")
    roc_ax.set_ylabel("True positive rate (fake caught)")
    roc_ax.set_title("Fake detection ROC (DeepFakeFace sample)")
    roc_ax.legend(loc="lower right", fontsize=8)
    roc_ax.grid(alpha=0.3)
    roc_fig.tight_layout()
    roc_fig.savefig(EVAL / "roc_detection.png", dpi=150)
    plt.close(roc_fig)
    out["plot"] = "data/eval/roc_detection.png"
    return out


# ---------------------------------------------------------------- report
def markdown(summary: dict) -> str:
    L = []
    v = summary.get("verification")
    if v:
        L += [f"### Verification: {v['pairs']}", "",
              "| Metric | Value |", "|---|---|",
              f"| ROC AUC | {v['roc_auc']:.4f} |",
              f"| EER | {v['eer'] * 100:.2f}% (t={v['eer_threshold']:.3f}) |",
              f"| FAR @ t_sim={v['t_sim']:.2f} | {v['far_at_t_sim'] * 100:.2f}% |",
              f"| FRR @ t_sim={v['t_sim']:.2f} | {v['frr_at_t_sim'] * 100:.2f}% |",
              f"| Accuracy @ t_sim | {v['accuracy_at_t_sim'] * 100:.2f}% |",
              f"| t_sim for FAR=1% | {v['t_sim_for_far_1pct']:.3f} (FRR {v['frr_at_t_far1'] * 100:.2f}%) |",
              f"| Mean sim genuine / impostor | {v['genuine_sim_mean']:.3f} / {v['impostor_sim_mean']:.3f} |",
              f"| Pairs with no face found | {v['failure_to_acquire_pairs']} |", ""]
    d = summary.get("detection")
    if d:
        L += [f"### Fake detection: {d['n_real']} real + {d['n_fake']} fake, t_fake={d['t_fake']:.2f}", "",
              "| Model | Acc | Prec | Recall | F1 | AUC | TN/FP/FN/TP | No face | Best t (acc) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for mid, r in d["models"].items():
            cm = r["confusion_matrix"]
            auc = f"{r['roc_auc']:.3f}" if r["roc_auc"] is not None else "n/a"
            L.append(f"| {mid} | {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                     f"{r['f1']:.3f} | {auc} | {cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']} | "
                     f"{r['no_face']} | {r['best_youden_threshold']:.3f} ({r['accuracy_at_best_threshold']:.3f}) |")
        L += ["", "Flagged-as-fake rate by source:", "",
              "| Model | " + " | ".join(next(iter(d["models"].values()))["per_source"]) + " |",
              "|---|" + "---|" * len(next(iter(d["models"].values()))["per_source"])]
        for mid, r in d["models"].items():
            L.append(f"| {short(mid)} | " + " | ".join(
                f"{s['flagged_fake_rate'] * 100:.0f}%" for s in r["per_source"].values()) + " |")
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="N pairs per class and N images per class")
    ap.add_argument("--detector", default=None, help="detector model id (default: config)")
    ap.add_argument("--compare", action="store_true", help="evaluate both candidate detectors")
    ap.add_argument("--pairs-file", default=None, help="pairs file in data/lfw (default pairs.txt)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--skip-verification", action="store_true")
    ap.add_argument("--skip-detection", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="delete embedding/detector caches first (use after changing ml/ code)")
    args = ap.parse_args()

    EVAL.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for p in [*EVAL.glob("embeddings_*.npz"), *EVAL.glob("detector_*.json")]:
            p.unlink()
            print(f"  removed cache {p.name}")
    primary = args.detector or settings.detector_model_id
    model_ids = [primary]
    if args.compare:
        for m in (settings.detector_model_id, settings.detector_fallback_id):
            if m not in model_ids:
                model_ids.append(m)
    models = Models(primary)

    summary: dict = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "limit": args.limit, "seed": args.seed,
        "thresholds": {"t_sim": settings.t_sim, "t_fake": settings.t_fake},
        "verifier": f"InsightFace {settings.insightface_pack} ArcFace",
        "detectors": model_ids,
    }
    if not args.skip_verification:
        summary["verification"] = verification(args, models)
    if not args.skip_detection:
        summary["detection"] = detection(args, models, model_ids)

    name = "summary" if not args.limit else f"summary_limit{args.limit}"
    (EVAL / f"{name}.json").write_text(json.dumps(summary, indent=2))
    md = markdown(summary)
    (EVAL / f"{name}.md").write_text(md)
    print()
    print(md)
    print(f"wrote {EVAL / (name + '.json')} and plots in {EVAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
