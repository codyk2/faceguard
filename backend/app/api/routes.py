"""HTTP endpoints (DESIGN.md section 4). Depends only on schemas, interfaces, store, config."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.deps import get_pipeline, get_store
from app.config import settings
from app.ml.interfaces import NoFaceError, Pipeline
from app.schemas import (
    DetectResponse,
    EnrollResponse,
    HealthResponse,
    Identity,
    RejectedImage,
    Thresholds,
    VerifyResponse,
)
from app.store import DuplicateNameError, Store

router = APIRouter()


def _thresholds() -> Thresholds:
    return Thresholds(t_sim=settings.t_sim, t_fake=settings.t_fake)


@router.get("/health", response_model=HealthResponse)
def health(pipeline: Pipeline = Depends(get_pipeline)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=pipeline is not None,
        device=getattr(pipeline, "device", "unknown"),
        verifier=getattr(pipeline, "verifier_name", "unknown"),
        detector=getattr(pipeline, "detector_name", "unknown"),
    )


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(
    name: str = Form(...),
    files: list[UploadFile] | None = File(None),
    pipeline: Pipeline = Depends(get_pipeline),
    store: Store = Depends(get_store),
) -> EnrollResponse:
    name = name.strip()
    if not name:
        raise HTTPException(400, "name must not be empty")
    files = files or []
    lo, hi = settings.enroll_min_images, settings.enroll_max_images
    if not lo <= len(files) <= hi:
        raise HTTPException(400, f"expected {lo}-{hi} images, got {len(files)}")
    if await run_in_threadpool(store.name_exists, name):
        raise HTTPException(409, f"identity '{name}' already exists")

    embeddings = []
    rejected: list[RejectedImage] = []
    for i, f in enumerate(files):
        data = await f.read()
        try:
            a = await run_in_threadpool(pipeline.analyze, data)
        except NoFaceError:
            rejected.append(RejectedImage(index=i, reason="no face detected"))
            continue
        except ValueError:
            rejected.append(RejectedImage(index=i, reason="invalid image"))
            continue
        if a.fake_probability >= settings.t_fake:
            rejected.append(
                RejectedImage(index=i, reason=f"flagged as fake (p={a.fake_probability:.2f})")
            )
            continue
        embeddings.append(a.embedding)

    if not embeddings:
        raise HTTPException(
            400,
            {
                "message": "no image accepted",
                "rejected": [r.model_dump() for r in rejected],
            },
        )
    try:
        ident = await run_in_threadpool(store.create_identity, name, embeddings)
    except DuplicateNameError:
        raise HTTPException(409, f"identity '{name}' already exists")
    return EnrollResponse(
        identity_id=ident["identity_id"],
        name=ident["name"],
        images_used=ident["images_used"],
        rejected=rejected,
    )


@router.get("/identities", response_model=list[Identity])
async def list_identities(store: Store = Depends(get_store)) -> list[Identity]:
    rows = await run_in_threadpool(store.list_identities)
    return [Identity(**r) for r in rows]


@router.delete("/identities/{identity_id}", status_code=204)
async def delete_identity(identity_id: int, store: Store = Depends(get_store)) -> Response:
    if not await run_in_threadpool(store.delete_identity, identity_id):
        raise HTTPException(404, "identity not found")
    return Response(status_code=204)


@router.post("/verify", response_model=VerifyResponse)
async def verify(
    identity_id: int = Form(...),
    file: UploadFile = File(...),
    pipeline: Pipeline = Depends(get_pipeline),
    store: Store = Depends(get_store),
) -> VerifyResponse:
    template = await run_in_threadpool(store.get_template, identity_id)
    if template is None:
        raise HTTPException(404, "identity not found")
    data = await file.read()
    try:
        r = await run_in_threadpool(pipeline.verify, data, template)
    except ValueError:
        raise HTTPException(400, "invalid image")
    return VerifyResponse(
        decision=r.decision,
        similarity=r.similarity,
        fake_probability=r.fake_probability,
        thresholds=_thresholds(),
        face_box=r.face_box,
        warnings=r.warnings,
        timings_ms=r.timings_ms,
    )


@router.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    pipeline: Pipeline = Depends(get_pipeline),
) -> DetectResponse:
    data = await file.read()
    try:
        a = await run_in_threadpool(pipeline.analyze, data)
    except NoFaceError:
        return DetectResponse(
            fake_probability=None, label=None, face_box=None, warnings=["no face detected"]
        )
    except ValueError:
        raise HTTPException(400, "invalid image")
    return DetectResponse(
        fake_probability=a.fake_probability,
        label=a.fake_label,
        face_box=a.face_box,
        warnings=a.warnings,
    )
