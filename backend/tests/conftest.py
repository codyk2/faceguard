from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.deps import FakePipeline
from app.main import create_app
from app.store import Store


def img(*markers: str) -> bytes:
    """Fake-pipeline image: PNG magic + steering markers (see app/api/deps.py)."""
    return b"\x89PNG\r\n\x1a\n" + b"".join(m.encode() for m in markers)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "sub" / "faceguard.db"


@pytest.fixture
def store(db_path) -> Store:
    return Store(db_path)


@pytest.fixture
def client(db_path):
    app = create_app(db_path=db_path, pipeline=FakePipeline())
    with TestClient(app) as c:  # context manager runs the lifespan
        yield c
