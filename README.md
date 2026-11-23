# FaceGuard

Face verification with deepfake screening. A person claims an enrolled identity and submits a face photo; the system accepts only if the photo is judged **real** and **matches** the enrolled identity.

| Component | Model |
|---|---|
| Face detection + alignment | SCRFD-10GF (InsightFace `buffalo_l`) |
| Verification | ArcFace ResNet50, 512-d embeddings, cosine similarity (InsightFace `buffalo_l`) |
| Fake detection | ViT-B/16 fine-tuned on DeepFakeFace from [`prithivMLmods/Deep-Fake-Detector-v2-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) (see [Fine-tuned detector](#fine-tuned-detector); without it the app falls back to the base model) |
| Backend | FastAPI, SQLite |
| Frontend | React + TypeScript + Vite |

See [DESIGN.md](DESIGN.md) for the full design and API contract.

## Pipeline

```
image -> SCRFD detect (once) -> main face -> aligned 112x112 -> ArcFace -> cosine vs enrolled template
                                          -> margin crop 224x224 -> ViT -> P(fake)

REJECT_FAKE if P(fake) >= T_FAKE, else REJECT_MISMATCH if cosine < T_SIM, else ACCEPT  (NO_FACE if no usable face)
```

## Quick start (macOS, Apple Silicon)

Requires [uv](https://docs.astral.sh/uv/), Node 20+, and Xcode Command Line Tools (InsightFace compiles a Cython extension).

```bash
make setup            # uv sync (Python 3.11) + npm install
make data             # LFW (13,233 imgs) + DeepFakeFace sample (300 real / 300 fake)
make inference-test   # runs both models, prints a results table
make dev              # backend :8000 + frontend :5173
make seed             # (backend running) enroll 10 LFW celebrities
make test             # fast tests; `make test-models` runs real-model tests
make evaluate         # FAR/FRR/ROC on LFW pairs, detector metrics on the fake set
```

Open http://localhost:5173. Frontend without a backend: `cd frontend && npm run dev:mock`.
Backend without models (for UI work): `FACEGUARD_FAKE_PIPELINE=1 make backend`.

Thresholds are set in `backend/app/config.py` or with env vars `FACEGUARD_T_SIM`, `FACEGUARD_T_FAKE`; the detector with `FACEGUARD_DETECTOR`.

## Data

- **LFW** — Huang et al., 2007. Real celebrity faces for enrollment and verification. Downloaded from scikit-learn's figshare mirror, SHA-256 verified.
- **DeepFakeFace (DFF)** — Song et al., arXiv:2309.02218, 2023. [OpenRL/DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace). 300 real (IMDB-WIKI) and 300 fake (100 SD inpainting, 100 InsightFace face swap, 100 SD v1.5 text-to-image), seed 1234. See `data/fakes/SOURCE.md` after download.

No group-member photos are collected, so no consent forms are needed.

## Fine-tuned detector

The off-the-shelf detectors score near chance on DeepFakeFace (AUC 0.54 and 0.50), so the ViT is fine-tuned:

```bash
make train-detector   # download training pairs, crop faces, train 4 epochs, evaluate (~25 min)
```

- **Data:** 1,500 DeepFakeFace source photos, each giving a real image plus one fake (rotating inpainting / face swap / text-to-image). No source photo of the 600-image evaluation set is used (asserted in `scripts/download_train.py`). 90/10 train/validation split by source photo.
- **Inputs:** face crops from the app's own `FacePreprocessor`, so training sees exactly what inference sees.
- **Shortcut control:** DeepFakeFace's inpainting and face-swap fakes were stretched to 512×512, so their faces are 12–30% wider than in the source photo. Crop shape alone gives AUC 0.68. Training applies random aspect stretch, resolution loss and JPEG re-compression to **both** classes.
- **Recipe:** AdamW, lr 2e-5, batch 32, 4 epochs, cosine schedule, label smoothing 0.05; best epoch by validation AUC. Output in `data/models/detector-dff-ft-lowres/` (not in git).
- **Threshold:** the fine-tuned model still gives LFW's old, low-resolution news photos higher P(fake) than DeepFakeFace reals. `T_FAKE = 0.90` is calibrated on half of a 1,000-image LFW sample (split by person) to about 5% false positives, and checked on the other half.

| Held-out DeepFakeFace test (469 crops) | Base | Fine-tuned |
|---|---|---|
| ROC AUC | 0.548 | **0.950** |
| Fakes caught (at about 5% real false positives) | 8% | **52%** |
| LFW real photos flagged at the app threshold | — | 4.8% |

End to end through the live API (T_SIM 0.35, T_FAKE 0.90): genuine users 95% ACCEPT, impostors 100% rejected, and DeepFakeFace fakes 47% REJECT_FAKE, 35% REJECT_MISMATCH, 18% NO_FACE. All 40 seed enrollment photos are accepted (22 of 40 with the base detector).

**Limits:** results are in-domain (same generators as training). Deepfakes *of an enrolled person* weren't tested. Choosing `FACEGUARD_T_FAKE=0.87` catches 63% of fakes at about 10% false positives on LFW.

## Dependency notes

Problems hit while setting up, and their fixes:

1. **iCloud-synced Desktop breaks virtualenvs** — the project lives in `~/Developer`, not `~/Desktop`.
2. **InsightFace build** — `insightface==0.7.3` ships as an sdist and builds a Cython extension; needs Xcode CLT. Python 3.11 used for best arm64 wheel coverage of `onnxruntime`/`torch`.
3. **NumPy 2 incompatibility** — pinned `numpy<2.0` for InsightFace.
4. **Albumentations SSL warning at import** (uv's standalone Python lacks the system CA store) — set `NO_ALBUMENTATIONS_UPDATE=1` (done in code). Same root cause broke `urllib` downloads; fixed with a `certifi` SSL context.
5. **InsightFace model location** — `buffalo_l.zip` (282 MB) downloads on first use to `data/models/models/buffalo_l/` (it nests `models/` under `root=`). Hugging Face weights go to `~/.cache/huggingface` (~343 MB each).
6. **CoreML execution provider** is skipped for InsightFace (forced CPU) for stability; ViT runs on MPS.
7. **transformers 5.x** works with the v4-style `AutoImageProcessor` / `AutoModelForImageClassification` API unchanged.
8. **scikit-image FutureWarning** from `face_align` (`SimilarityTransform.estimate` deprecated) — harmless.
9. **Detector label order differs between models** (`Realism/Deepfake` vs `Real/Fake`) — P(fake) index is found from `model.config.id2label`, never hard-coded.
10. **Face selection bug** — raw SCRFD boxes for faces cut off at the image edge extend past it and beat the centred subject on LFW crops. Fixed by ranking faces by image-clipped area weighted toward the centre.

## Labor division

Split 50/50; modality and model choices and end-to-end testing were joint.

| Member | Responsibilities |
|---|---|
| Cody Kandarian | Verification (ArcFace) integration; face detection, alignment and main-face selection; FastAPI backend and SQLite storage; environment and dependency fixes; LFW data, seed enrollment and verification evaluation; GitHub repository |
| Raleigh Tognela | Deepfake detector selection and integration; DeepFakeFace data, detector fine-tuning and threshold calibration; detection evaluation; React web interface; inference-test screenshots and report |

## Citations

- Deng, Guo, Xue, Zafeiriou. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019.
- Guo, Deng, Lattas, Zafeiriou. Sample and Computation Redistribution for Efficient Face Detection (SCRFD). ICLR 2022.
- Dosovitskiy et al. An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
- prithivMLmods. Deep-Fake-Detector-v2-Model. Hugging Face model card, accessed 2026-11-21.
- Huang, Ramesh, Berg, Learned-Miller. Labeled Faces in the Wild. UMass Amherst TR 07-49, 2007.
- Song, Huang, Dong, Tu. Robustness and Generalizability of Deepfake Detection: A Study with Diffusion Models. arXiv:2309.02218, 2023.
