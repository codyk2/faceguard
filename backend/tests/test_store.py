from __future__ import annotations

import numpy as np
import pytest

from app.ml.interfaces import make_template
from app.store import DuplicateNameError, Store


def _vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_creates_parent_dir(db_path):
    assert not db_path.parent.exists()
    Store(db_path)
    assert db_path.exists()


def test_create_list_template_delete(store):
    embs = [_vec(1), _vec(2), _vec(3)]
    ident = store.create_identity("alice", embs)
    assert ident["images_used"] == 3

    rows = store.list_identities()
    assert [(r["name"], r["images_used"]) for r in rows] == [("alice", 3)]

    t = store.get_template(ident["identity_id"])
    assert t.dtype == np.float32 and t.shape == (512,)
    np.testing.assert_allclose(t, make_template(embs), rtol=1e-5, atol=1e-6)
    assert abs(np.linalg.norm(t) - 1) < 1e-5

    assert store.delete_identity(ident["identity_id"]) is True
    assert store.list_identities() == []
    assert store.get_embeddings(ident["identity_id"]) == []  # cascade
    assert store.get_template(ident["identity_id"]) is None
    assert store.delete_identity(ident["identity_id"]) is False


def test_duplicate_name(store):
    store.create_identity("bob", [_vec(1)])
    assert store.name_exists("bob")
    with pytest.raises(DuplicateNameError):
        store.create_identity("bob", [_vec(2)])
    assert len(store.list_identities()) == 1


def test_bad_embedding_shape_rolls_back(store):
    with pytest.raises(ValueError):
        store.create_identity("carol", [np.zeros(128, dtype=np.float32)])
    assert store.list_identities() == []


def test_persists_across_instances(db_path):
    Store(db_path).create_identity("dave", [_vec(5)])
    assert Store(db_path).list_identities()[0]["name"] == "dave"
