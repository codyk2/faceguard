"""API contract. FROZEN: frontend/src/api.ts mirrors these types exactly."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT_FAKE = "REJECT_FAKE"
    REJECT_MISMATCH = "REJECT_MISMATCH"
    NO_FACE = "NO_FACE"


class FaceBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


class Thresholds(BaseModel):
    t_sim: float
    t_fake: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str
    verifier: str
    detector: str


class RejectedImage(BaseModel):
    index: int
    reason: str


class EnrollResponse(BaseModel):
    identity_id: int
    name: str
    images_used: int
    rejected: list[RejectedImage]


class Identity(BaseModel):
    identity_id: int
    name: str
    images_used: int
    created_at: datetime


class VerifyResponse(BaseModel):
    decision: Decision
    similarity: float | None
    fake_probability: float | None
    thresholds: Thresholds
    face_box: FaceBox | None
    warnings: list[str]
    timings_ms: dict[str, float]


class DetectResponse(BaseModel):
    fake_probability: float | None
    label: str | None
    face_box: FaceBox | None
    warnings: list[str]
