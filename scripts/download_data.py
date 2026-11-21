"""Download LFW and a real-vs-fake face sample into data/.

Usage (from backend/):
    uv run python ../scripts/download_data.py            # both
    uv run python ../scripts/download_data.py --only lfw
    uv run python ../scripts/download_data.py --only fakes --n-real 300 --n-fake 300

Idempotent: anything already present is skipped. Downloaded archives are treated
as untrusted data: only regular .jpg members with safe relative paths are
extracted, each is decoded with Pillow before being kept, nothing is executed.

Layout produced:
    data/lfw/<Person>/<Person>_0001.jpg   raw 250x250 colour LFW images
    data/lfw/pairs.txt, pairsDevTest.txt, pairsDevTrain.txt
    data/fakes/real/*.jpg, data/fakes/fake/*.jpg, data/fakes/SOURCE.md,
    data/fakes/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import shutil
import ssl
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import _common as C
from PIL import Image

# --- LFW (same mirror + checksums scikit-learn's fetch_lfw_people uses) ------
LFW_ARCHIVE = (
    "lfw.tgz",
    "https://ndownloader.figshare.com/files/5976018",
    "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0",
)
LFW_META = [
    ("pairs.txt", "https://ndownloader.figshare.com/files/5976006",
     "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"),
    ("pairsDevTest.txt", "https://ndownloader.figshare.com/files/5976009",
     "7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c"),
    ("pairsDevTrain.txt", "https://ndownloader.figshare.com/files/5976012",
     "1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa"),
]
LFW_EXPECTED_IMAGES = 13233
LFW_EXPECTED_PEOPLE = 5749

# --- Fakes: DeepFakeFace (DFF), OpenRL/DeepFakeFace on Hugging Face ----------
DFF_REPO = "OpenRL/DeepFakeFace"
DFF_REAL_ZIP = "wiki.zip"  # real photos selected from IMDB-WIKI
DFF_FAKE_ZIPS = ["inpainting.zip", "insight.zip", "text2img.zip"]
DFF_FAKE_DESC = {
    "inpainting.zip": "Stable Diffusion Inpainting (face region regenerated)",
    "insight.zip": "InsightFace face swap",
    "text2img.zip": "Stable Diffusion v1.5 text-to-image",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def download(url: str, dest: Path, sha256: str | None = None) -> None:
    if dest.exists() and (sha256 is None or C.sha256_file(dest) == sha256):
        log(f"  skip {dest.name} (present)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"  downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "faceguard-data/0.1"})
    # uv's standalone Python has no system CA store; use certifi's bundle
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    if sha256 is not None:
        got = C.sha256_file(tmp)
        if got != sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {dest.name}: {got} != {sha256}")
    tmp.replace(dest)


def _safe_member(name: str) -> bool:
    p = Path(name)
    return not p.is_absolute() and ".." not in p.parts and name.lower().endswith(".jpg")


def _is_jpeg(data: bytes) -> tuple[bool, tuple[int, int] | None, str | None]:
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, size, mode = im.format, im.size, im.mode
            im.load()  # full decode catches truncated files
        return fmt == "JPEG", size, mode
    except Exception:
        return False, None, None


# ----------------------------------------------------------------------------
def lfw_counts() -> tuple[int, int]:
    people = C.lfw_people()
    return sum(len(v) for v in people.values()), len(people)


def get_lfw() -> dict:
    log("[lfw]")
    lfw = C.LFW_DIR
    lfw.mkdir(parents=True, exist_ok=True)
    for fname, url, sha in LFW_META:
        try:
            download(url, lfw / fname, sha)
        except Exception as e:  # pairs are nice-to-have; evaluate.py can sample
            log(f"  WARNING could not fetch {fname}: {e}")

    n_img, n_ppl = lfw_counts()
    if n_img >= LFW_EXPECTED_IMAGES and n_ppl >= LFW_EXPECTED_PEOPLE:
        log(f"  skip images ({n_img} images / {n_ppl} people already extracted)")
    else:
        fname, url, sha = LFW_ARCHIVE
        archive = C.RAW_DIR / fname
        download(url, archive, sha)
        log(f"  extracting jpgs from {archive.name}")
        n = 0
        with tarfile.open(archive, "r:gz") as tf:
            for m in tf:
                # only regular files lfw/<Person>/<Person>_NNNN.jpg
                if not m.isreg() or not _safe_member(m.name):
                    continue
                parts = Path(m.name).parts
                if len(parts) != 3 or parts[0] != "lfw":
                    continue
                person, base = parts[1], parts[2]
                dest = lfw / person / base
                if dest.exists():
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                ok, _, _ = _is_jpeg(data)
                if not ok:
                    log(f"  skip non-jpeg member {m.name}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                n += 1
        log(f"  extracted {n} new images")
        n_img, n_ppl = lfw_counts()

    # integrity: counts + a spot check of image size/mode
    sample = next(iter(C.lfw_people().get("George_W_Bush", [])), None)
    size = mode = None
    if sample:
        with Image.open(sample) as im:
            size, mode = im.size, im.mode
    ok = n_img == LFW_EXPECTED_IMAGES and n_ppl == LFW_EXPECTED_PEOPLE
    pairs = [f for f, _, _ in LFW_META if (lfw / f).exists()]
    log(f"  LFW: {n_img} images, {n_ppl} people "
        f"(expected {LFW_EXPECTED_IMAGES}/{LFW_EXPECTED_PEOPLE}) -> {'OK' if ok else 'MISMATCH'}; "
        f"sample {size} {mode}; pair files: {pairs}")
    return {"images": n_img, "people": n_ppl, "ok": ok, "pairs_files": pairs,
            "sample_size": size, "sample_mode": mode}


# ----------------------------------------------------------------------------
def _open_remote_zip(fs, name: str) -> zipfile.ZipFile:
    f = fs.open(f"datasets/{DFF_REPO}/{name}", "rb", block_size=1 << 20)
    return zipfile.ZipFile(f)


def get_fakes(n_real: int, n_fake: int, seed: int) -> dict:
    """Sample DFF images by HTTP range reads into the remote zips (no 5 GB download)."""
    log("[fakes]")
    real_dir, fake_dir = C.FAKES_DIR / "real", C.FAKES_DIR / "fake"
    have_r = len(list(real_dir.glob("*.jpg"))) if real_dir.exists() else 0
    have_f = len(list(fake_dir.glob("*.jpg"))) if fake_dir.exists() else 0
    manifest_path = C.FAKES_DIR / "manifest.csv"
    if have_r >= n_real and have_f >= n_fake and manifest_path.exists():
        log(f"  skip ({have_r} real / {have_f} fake already present)")
        write_source_md(have_r, have_f, seed, _fake_breakdown(manifest_path))
        return {"real": have_r, "fake": have_f, "ok": True}

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    real_zip = _open_remote_zip(fs, DFF_REAL_ZIP)
    # member names are "<zip-stem>/<dir>/<id>_<dob>_<year>.jpg" and identical
    # across the 4 zips (fakes derive from the same IMDB-WIKI source photo).
    rel_names = sorted(
        n.split("/", 1)[1] for n in real_zip.namelist()
        if _safe_member(n) and n.count("/") == 2
    )
    log(f"  {len(rel_names)} source ids in {DFF_REAL_ZIP}")
    rng = random.Random(seed)
    order = rel_names[:]
    rng.shuffle(order)
    # disjoint source ids: real images never share a source photo with a fake,
    # so the detector can't be scored on a real/fake pair of the same picture.
    real_pool = order[: len(order) // 2]
    fake_pool = order[len(order) // 2:]

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def take(zf: zipfile.ZipFile, zip_name: str, pool: list[str], k: int,
             out_dir: Path, label: str) -> int:
        stem = zip_name.rsplit(".", 1)[0]
        got = 0
        for rel in pool:
            if got >= k:
                break
            out_name = f"{stem}_{rel.replace('/', '_')}"
            dest = out_dir / out_name
            member = f"{stem}/{rel}"
            if not dest.exists():
                try:
                    data = zf.read(member)
                except KeyError:
                    continue
                ok, size, mode = _is_jpeg(data)
                if not ok or size is None or min(size) < 64:
                    continue
                dest.write_bytes(data)
            else:
                with Image.open(dest) as im:
                    size, mode = im.size, im.mode
            rows.append({"file": f"{label}/{out_name}", "label": label, "source_zip": zip_name,
                         "member": member, "width": size[0], "height": size[1], "mode": mode})
            got += 1
        log(f"  {label}: {got} from {zip_name}")
        return got

    n_r = take(real_zip, DFF_REAL_ZIP, real_pool, n_real, real_dir, "real")
    # split fakes evenly across the three generators
    per = [n_fake // len(DFF_FAKE_ZIPS)] * len(DFF_FAKE_ZIPS)
    for i in range(n_fake - sum(per)):
        per[i] += 1
    n_f = 0
    offset = 0
    for zname, k in zip(DFF_FAKE_ZIPS, per):
        zf = _open_remote_zip(fs, zname)
        # each generator draws from its own slice of the fake pool -> distinct ids
        n_f += take(zf, zname, fake_pool[offset:], k, fake_dir, "fake")
        offset += len(fake_pool) // len(DFF_FAKE_ZIPS)

    # drop any stale files not in this sample (e.g. from a different --n-*)
    keep = {r["file"] for r in rows}
    for d, lab in ((real_dir, "real"), (fake_dir, "fake")):
        for p in d.glob("*.jpg"):
            if f"{lab}/{p.name}" not in keep:
                p.unlink()

    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_source_md(n_r, n_f, seed, _fake_breakdown(manifest_path))
    log(f"  fakes: {n_r} real, {n_f} fake -> {C.FAKES_DIR}")
    return {"real": n_r, "fake": n_f, "ok": n_r == n_real and n_f == n_fake}


def _fake_breakdown(manifest_path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(manifest_path) as f:
        for r in csv.DictReader(f):
            out[r["source_zip"]] = out.get(r["source_zip"], 0) + 1
    return out


def write_source_md(n_real: int, n_fake: int, seed: int, breakdown: dict[str, int]) -> None:
    lines = "\n".join(
        f"| `{z}` | {'real' if z == DFF_REAL_ZIP else 'fake'} | "
        f"{'IMDB-WIKI photo (real)' if z == DFF_REAL_ZIP else DFF_FAKE_DESC.get(z, '')} | {n} |"
        for z, n in breakdown.items()
    )
    md = f"""# Fake-detection evaluation set: source

**Dataset:** DeepFakeFace (DFF)
**URL:** https://huggingface.co/datasets/{DFF_REPO}
(project site https://sites.google.com/view/deepfakeface/home, code https://github.com/OpenRL-Lab/DeepFakeFace)
**License:** `openrail` per the Hugging Face dataset metadata (the README body also shows an
`apache-2.0` block). The real images are a subset of IMDB-WIKI, which its authors release for
academic research use only; use here is non-commercial coursework.
**Access:** public, not gated, no request form.

**Citation**

> Haixu Song, Shiyu Huang, Yinpeng Dong, Wei-Wei Tu. *Robustness and Generalizability of
> Deepfake Detection: A Study with Diffusion Models.* arXiv:2309.02218, 2023.

Real-image source: Rasmus Rothe, Radu Timofte, Luc Van Gool. *Deep Expectation of Real and
Apparent Age from a Single Image Without Facial Landmarks.* IJCV 2018 (IMDB-WIKI).

## What DFF contains

Four zips of 30,000 images each, sharing the IMDB-WIKI directory structure:
`wiki.zip` (real celebrity photos), `inpainting.zip` (Stable Diffusion Inpainting),
`insight.zip` (InsightFace face swap), `text2img.zip` (Stable Diffusion v1.5 text-to-image).

## Sample in this folder

| Zip | Class | Generator | Images |
|---|---|---|---|
{lines}

Totals: **{n_real} real** (`real/`), **{n_fake} fake** (`fake/`). Per-file provenance is in
`manifest.csv` (zip, member path, size).

## How files were sampled

`scripts/download_data.py --only fakes` lists the member names of `wiki.zip`, shuffles them
with `random.Random({seed})`, and splits the shuffled list in half. Real images come from the
first half (taken from `wiki.zip`); fakes come from the second half, split into three disjoint
slices, one per generator zip, with fakes divided evenly across the three generators. So no
real image shares a source photo or source ID with any fake, and no two fakes share a source
ID. Members are read with HTTP range requests into the remote zips (`huggingface_hub.HfFileSystem`),
so the 5 GB of archives are never fully downloaded. Each image is fully decoded with Pillow
and kept only if it is a valid JPEG with a short side of at least 64 px; original bytes are
stored unchanged (no re-encoding, so compression artefacts are untouched).

## Caveat: image-format confound

All DFF fakes are 512x512, while the real IMDB-WIKI photos vary in size (mostly 440 px wide)
and in JPEG history. The pipeline scores a face crop resized to 224x224, which weakens but
does not remove this cue. Keep it in mind when reading the detection numbers.

## Overlap with the detector models' training data

- `prithivMLmods/Deep-Fake-Detector-v2-Model`: the card says only "a curated dataset of real
  and deepfake images" (56,001 test images, labels Realism/Deepfake). The author's related HF
  datasets (Deepfake-vs-Real, -v2, -60K) say they are built from his Deepfakes-QA-Patch1/2
  sets, whose original sources are not stated. DFF is not named anywhere.
- `dima806/deepfake_vs_real_image_detection`: the card points to a Kaggle notebook
  (deepfake-vs-real-faces-detection-vit) and says the data was "collected about 3 years ago".
  The notebook's input dataset could not be read without a Kaggle login. DFF is not named.

So no overlap is documented, but since neither card fully discloses its training sources,
overlap cannot be ruled out entirely.
"""
    C.FAKES_DIR.mkdir(parents=True, exist_ok=True)
    (C.FAKES_DIR / "SOURCE.md").write_text(md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["lfw", "fakes"], help="fetch only one dataset")
    ap.add_argument("--n-real", type=int, default=300)
    ap.add_argument("--n-fake", type=int, default=300)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    summary: dict[str, dict] = {}
    if args.only in (None, "lfw"):
        summary["lfw"] = get_lfw()
    if args.only in (None, "fakes"):
        summary["fakes"] = get_fakes(args.n_real, args.n_fake, args.seed)

    log("\n== summary ==")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    return 0 if all(v.get("ok") for v in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
