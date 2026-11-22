# FaceGuard — design

Face verification with deepfake screening. A person claims an enrolled identity and submits a face image; the system accepts only if the image is judged real **and** it matches the enrolled identity.

Status: design, nothing built yet. Milestone 1 (Setup and Model Selection, 15 pts) is due Nov 22, one grace day max.

## 1. Decisions

| Area | Choice | Notes |
|---|---|---|
| Modality | Face | |
| Verification | InsightFace `buffalo_l` | SCRFD-10GF face detector + ArcFace ResNet50 embeddings (512-d), run through ONNX Runtime. Weights are licensed for non-commercial research, which covers coursework. |
| Fake detection | ViT deepfake classifier | `prithivMLmods/Deep-Fake-Detector-v2-Model` (primary) or `dima806/deepfake_vs_real_image_detection` (fallback). Both are `google/vit-base-patch16-224-in21k` fine-tunes, Apache-2.0, confirmed present on Hugging Face on 2026-11-21. |
| Real faces | Public celebrity photos | LFW (Labeled Faces in the Wild) through `sklearn.datasets.fetch_lfw_people`. No consent forms needed since no one in the group is photographed. |
| Fake faces | Public benchmark samples | Open item, see section 9. |
| Backend | FastAPI, Python 3.11, `uv` | |
| Frontend | React + TypeScript + Vite | |
| Storage | SQLite + embeddings as BLOBs | Single file, no server. |

## 2. Pipeline

```
image ──> decode ──> SCRFD detect ──> pick largest face ──┬─> 5-pt align 112x112 ──> ArcFace ──> 512-d embedding ──> cosine vs template
                                                          │
                                                          └─> bbox crop +30% margin, 224x224 ──> ViT ──> P(fake)

decision = ACCEPT  if  P(fake) < T_FAKE  and  cosine >= T_SIM
           REJECT_FAKE      if  P(fake) >= T_FAKE        (checked first)
           REJECT_MISMATCH  if  cosine < T_SIM
           NO_FACE          if  no face detected
```

Rules the builders must follow:

- **One detection pass feeds both models.** The detector runs once; the verifier gets the aligned crop, the ViT gets the margin crop.
- **Enrollment template** = mean of the L2-normalized embeddings of 3–5 images, re-normalized. Enrollment images also pass through the fake detector; a fake is refused.
- **Thresholds are config, not constants.** Start with `T_SIM = 0.35` and `T_FAKE = 0.5`, then calibrate on LFW pairs and the fake set (section 7). Both live in `backend/app/config.py` and are echoed in every API response.
- **Read the ViT label order from `model.config.id2label`.** Never hard-code which index means fake; the two candidate models may differ.
- Multiple faces: use the largest and return a warning. Face smaller than 60 px on its short side: reject as low quality.

## 3. Repository layout

Location: **outside `~/Desktop`** (Desktop is iCloud-synced, which corrupts virtualenvs and creates duplicate files). Default: `~/Developer/faceguard`.

```
faceguard/
  README.md  Makefile  .gitignore  .python-version (3.11)
  backend/
    pyproject.toml
    app/
      main.py            FastAPI app, CORS, router mount
      config.py          thresholds, model ids, paths
      schemas.py         Pydantic request/response models (the contract)
      store.py           SQLite: identities, embeddings
      api/routes.py      endpoints
      ml/preprocess.py   decode, detect, align, crop
      ml/verifier.py     ArcFace embed + cosine
      ml/detector.py     ViT P(fake)
      ml/pipeline.py     decision logic
    tests/
  frontend/              Vite + React + TS
    src/api.ts           typed client for schemas.py
    src/pages/           Enroll.tsx  Verify.tsx  Identities.tsx
    src/components/      ImageInput.tsx (webcam + upload)  ResultCard.tsx
  scripts/
    download_data.py     LFW + fake samples into data/
    inference_test.py    runs both models, prints results (the screenshot source)
    seed_enroll.py       enrolls N celebrities from LFW
    evaluate.py          FAR/FRR, ROC, threshold sweep
  data/                  gitignored
  report/                milestone PDF source + screenshots
```

## 4. API contract

All under `/api`. Images are `multipart/form-data`.

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/health` | | `{status, models_loaded, device}` |
| POST | `/enroll` | `name`, `files[]` (3–5) | `{identity_id, name, images_used, rejected: [{index, reason}]}` |
| GET | `/identities` | | `[{identity_id, name, images_used, created_at}]` |
| DELETE | `/identities/{id}` | | `204` |
| POST | `/verify` | `identity_id`, `file` | `{decision, similarity, fake_probability, thresholds, face_box, warnings, timings_ms}` |
| POST | `/detect` | `file` | `{fake_probability, label, face_box}` |

`decision` is one of `ACCEPT`, `REJECT_FAKE`, `REJECT_MISMATCH`, `NO_FACE`. Models load once at startup (FastAPI lifespan), not per request.

## 5. Frontend

Three pages, one shared `ImageInput` that offers webcam capture or file upload.

- **Enroll**: name + 3–5 images; shows per-image accept/reject with reason.
- **Verify**: pick identity, submit one image; `ResultCard` shows the decision, a similarity bar against `T_SIM`, a fake-probability bar against `T_FAKE`, and the detected face box drawn on the image.
- **Identities**: list and delete.

Vite dev server proxies `/api` to `localhost:8000`.

## 6. Environment

- Python 3.11 through `uv` (installed). InsightFace compiles a Cython extension, so Xcode Command Line Tools are required; 3.11 has the best wheel coverage for `onnxruntime` and `torch` on arm64.
- Key packages: `insightface`, `onnxruntime`, `torch`, `transformers`, `pillow`, `opencv-python-headless`, `numpy`, `scikit-learn`, `fastapi`, `uvicorn`, `python-multipart`, `pytest`, `httpx`.
- ArcFace on ONNX Runtime CPU; ViT on PyTorch with MPS if available, CPU otherwise.
- `make setup`, `make dev` (backend + frontend), `make test`, `make inference-test`.
- Record every dependency problem hit and its fix in the README; the course expects this to be the hard part.

## 7. Testing

- **Unit**: preprocess (face found / none / multiple / too small), cosine and template maths, decision table for all four outcomes with mocked scores, label mapping read from config.
- **Model smoke** (`inference_test.py`): same-person LFW pair scores above different-person pair; real sample scores lower P(fake) than a fake sample. Prints a small table; this is the screenshot for the milestone.
- **API**: `httpx` tests for enroll → verify → delete, bad file, unknown identity, no-face image.
- **End to end**: seed enroll, then in the browser verify a genuine photo, an impostor photo and a fake; expect ACCEPT, REJECT_MISMATCH, REJECT_FAKE.
- **Evaluation** (later milestone): FAR/FRR and ROC on LFW pairs, detector accuracy on the fake set, threshold sweep.

## 8. Build plan for the agent swarm

Phase 0 runs alone and commits; phase 1 packages touch disjoint files and run in parallel; phase 2 runs last.

| Phase | Package | Owns | Done when |
|---|---|---|---|
| 0 | Scaffold | root files, `pyproject.toml`, `config.py`, `schemas.py`, frontend skeleton | `make setup` works; contract in `schemas.py` and `src/api.ts` is frozen |
| 1 | A. ML core | `backend/app/ml/`, `scripts/inference_test.py` | `make inference-test` prints sensible scores for both models |
| 1 | B. API + storage | `main.py`, `store.py`, `api/`, API tests | API tests pass against a mocked pipeline |
| 1 | C. Frontend | `frontend/` | all three pages work against a mocked API |
| 1 | D. Data + eval | `scripts/download_data.py`, `seed_enroll.py`, `evaluate.py` | data downloads reproducibly; 10 celebrities enrolled |
| 2 | Integration + QA | tests, fixes across packages | section 7 end-to-end passes with real models |
| 2 | Report | `report/` | PDF covering the four milestone items |

## 9. Milestone 1 mapping (due Nov 22)

| Requirement | Points | Source |
|---|---|---|
| Modality choice | | Face |
| Models with citations + 1–2 paragraphs on how the detector works | 5 | Draft below |
| Inference test screenshots, both models | 5 | `make inference-test` output + one UI screenshot |
| GitHub repo with environment; labor division | 5 | Repo from phase 0; labor division needs group names |

**How the detector works (draft).** The detector is a Vision Transformer (ViT-Base/16) pretrained on ImageNet-21k and fine-tuned as a binary real-versus-fake face classifier. The face crop is resized to 224×224 and cut into 196 non-overlapping 16×16 patches. Each patch is linearly projected to a 768-dimensional token, a position embedding is added, and a learned classification token is prepended. Twelve transformer encoder layers then apply multi-head self-attention, which lets every patch attend to every other patch.

That global attention is what suits the task: generated and face-swapped images tend to carry inconsistencies that span regions, such as blending seams at the face boundary, mismatched texture or lighting between the face and its surroundings, and unnatural high-frequency detail. The final state of the classification token goes through a linear head and softmax to give P(fake), which we threshold. A known limitation is that such classifiers generalize poorly to generation methods absent from their fine-tuning data, so we report accuracy on our own fake samples rather than relying on the model card.

**Citations**

- Deng, Guo, Xue, Zafeiriou. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019.
- Guo, Deng, Lattas, Zafeiriou. *Sample and Computation Redistribution for Efficient Face Detection* (SCRFD). ICLR 2022.
- Dosovitskiy et al. *An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
- Hugging Face model card for the chosen detector (author, URL, access date).
- Huang, Ramesh, Berg, Learned-Miller. *Labeled Faces in the Wild.* UMass Amherst Tech Report 07-49, 2007.

## 10. Open items

Resolved on 2026-11-21:

- **Fake samples:** an open real-vs-fake face image dataset from Hugging Face or Kaggle that downloads without a request form. The data agent picks one, checks its license, and records name, URL and citation in the README. Celeb-DF v2 is out of scope for now.
- **Group:** Cody Kandarian and Raleigh, split 50/50. Cody: verification, preprocessing, backend, environment, LFW evaluation. Raleigh: detection, fine-tuning, frontend, detection evaluation, report.
- **Repo:** `~/Developer/faceguard`, pushed to a private GitHub repo `codyk2/faceguard`. The grader needs a collaborator invite or the repo made public before submission.

Still open:

- **Risk:** the ViT detector may score poorly on fakes from generators it was not trained on. If the smoke test shows this, switch to the fallback model before the screenshots are taken.
