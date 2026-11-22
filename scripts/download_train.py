"""Download a DeepFakeFace TRAINING set for fine-tuning the fake detector.

Disjoint from the evaluation set in data/fakes/: every IMDB-WIKI source photo id used by
any of the 600 evaluation images is excluded, so no test picture (real or any of its
fakes) is ever trained on.

Training samples are PAIRED: each chosen source id contributes its real photo (wiki.zip)
and one fake derived from it, rotating inpainting -> insight -> text2img. Pairs make the
classifier learn generation artifacts instead of who or what is in the picture.

Output: data/fakes_train/{real,fake}/*.jpg and data/fakes_train/manifest.csv
(columns: file, label, source_zip, member, source_id, width, height).

Usage (from backend/):  uv run python ../scripts/download_train.py [--pairs 1500] [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import _common as C
from download_data import (DFF_FAKE_ZIPS, DFF_REAL_ZIP, _is_jpeg, _open_remote_zip,
                           _safe_member)

TRAIN_DIR = C.DATA_DIR / "fakes_train"
TRAIN_SEED = 2026


def test_source_ids() -> set[str]:
    """rel paths ('<dir>/<id>_<dob>_<year>.jpg') of every evaluation image."""
    ids = set()
    with open(C.FAKES_DIR / "manifest.csv") as f:
        for r in csv.DictReader(f):
            ids.add(r["member"].split("/", 1)[1])
    return ids


_local = threading.local()


def _zip(fs, name):
    """One remote ZipFile handle per (thread, zip) - zipfile handles aren't thread-safe."""
    cache = getattr(_local, "zips", None)
    if cache is None:
        cache = _local.zips = {}
    if name not in cache:
        cache[name] = _open_remote_zip(fs, name)
    return cache[name]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    real_zip = _open_remote_zip(fs, DFF_REAL_ZIP)
    rel_names = sorted(n.split("/", 1)[1] for n in real_zip.namelist()
                       if _safe_member(n) and n.count("/") == 2)
    excluded = test_source_ids()
    pool = [r for r in rel_names if r not in excluded]
    print(f"{len(rel_names)} source ids, {len(excluded)} used by eval set -> {len(pool)} available", flush=True)
    rng = random.Random(TRAIN_SEED)
    rng.shuffle(pool)

    (TRAIN_DIR / "real").mkdir(parents=True, exist_ok=True)
    (TRAIN_DIR / "fake").mkdir(parents=True, exist_ok=True)

    # Over-provision candidates: some members are missing or unreadable in a fake zip.
    candidates = pool[: int(args.pairs * 1.15)]
    jobs = []  # (idx, rel, fake_zip)
    for i, rel in enumerate(candidates):
        jobs.append((i, rel, DFF_FAKE_ZIPS[i % len(DFF_FAKE_ZIPS)]))

    def fetch(job):
        i, rel, fzip = job
        out = []
        for zname, label in ((DFF_REAL_ZIP, "real"), (fzip, "fake")):
            stem = zname.rsplit(".", 1)[0]
            member = f"{stem}/{rel}"
            dest = TRAIN_DIR / label / f"{stem}_{rel.replace('/', '_')}"
            if dest.exists():
                data = dest.read_bytes()
            else:
                try:
                    data = _zip(fs, zname).read(member)
                except KeyError:
                    return None
            ok, size, _mode = _is_jpeg(data)
            if not ok or size is None or min(size) < 64:
                return None
            out.append((dest, data, {"file": f"{label}/{dest.name}", "label": label,
                                     "source_zip": zname, "member": member,
                                     "source_id": rel, "width": size[0], "height": size[1]}))
        # write the pair immediately so an interrupted run can resume from disk
        for dest, data, _row in out:
            if not dest.exists():
                dest.write_bytes(data)
        return i, out

    results: dict[int, list] = {}
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res:
                results[res[0]] = res[1]
            if n % 200 == 0:
                print(f"  {n}/{len(jobs)} fetched ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    kept = 0
    for i in sorted(results):
        if kept >= args.pairs:
            break
        for dest, data, row in results[i]:
            if not dest.exists():
                dest.write_bytes(data)
            rows.append(row)
        kept += 1
    # remove files from over-provisioned candidates that weren't kept
    keep = {r["file"] for r in rows}
    for lab in ("real", "fake"):
        for p in (TRAIN_DIR / lab).glob("*.jpg"):
            if f"{lab}/{p.name}" not in keep:
                p.unlink()

    with open(TRAIN_DIR / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    assert not ({r["source_id"] for r in rows} & excluded), "leak: training id in eval set"
    by_zip: dict[str, int] = {}
    for r in rows:
        by_zip[r["source_zip"]] = by_zip.get(r["source_zip"], 0) + 1
    print(f"kept {kept} pairs in {time.time() - t0:.0f}s: {by_zip}")
    print("overlap with eval source ids: 0 (checked)")
    return 0 if kept == args.pairs else 1


if __name__ == "__main__":
    sys.exit(main())
