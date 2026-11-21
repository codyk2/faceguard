"""FastAPI entrypoint: `uvicorn app.main:app`.

Set FACEGUARD_FAKE_PIPELINE=1 to run without models (deterministic fake, see app/api/deps.py).
"""
from __future__ import annotations

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import logging  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.api.routes import router  # noqa: E402
from app.ml.interfaces import Pipeline  # noqa: E402
from app.store import Store  # noqa: E402

log = logging.getLogger("faceguard")


def _use_fake() -> bool:
    return os.getenv("FACEGUARD_FAKE_PIPELINE", "").lower() in {"1", "true", "yes"}


def create_app(db_path: Path | str | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = Store(db_path)
        if pipeline is not None:
            app.state.pipeline = pipeline
        elif _use_fake():
            from app.api.deps import FakePipeline

            log.warning("FACEGUARD_FAKE_PIPELINE set: using deterministic fake pipeline")
            app.state.pipeline = FakePipeline()
        else:
            from app.ml.pipeline import load_pipeline

            app.state.pipeline = load_pipeline()
        yield

    app = FastAPI(title="FaceGuard", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")
    return app


app = create_app()
