from __future__ import annotations

from app.config import settings
from tests.conftest import img


def _files(*blobs: bytes):
    return [("files", (f"img{i}.png", b, "image/png")) for i, b in enumerate(blobs)]


def _enroll(client, name="alice", person="alice", n=3):
    return client.post(
        "/api/enroll",
        data={"name": name},
        files=_files(*[img(f"FG:PERSON={person};", f"#{i}") for i in range(n)]),
    )


def _verify(client, identity_id, blob):
    return client.post(
        "/api/verify",
        data={"identity_id": str(identity_id)},
        files={"file": ("probe.png", blob, "image/png")},
    )


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["models_loaded"] is True
    assert {"device", "verifier", "detector"} <= body.keys()


def test_full_flow(client):
    r = _enroll(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "alice" and body["images_used"] == 3 and body["rejected"] == []
    iid = body["identity_id"]

    ids = client.get("/api/identities").json()
    assert len(ids) == 1
    assert ids[0]["identity_id"] == iid and ids[0]["images_used"] == 3
    assert "created_at" in ids[0]

    r = _verify(client, iid, img("FG:PERSON=alice;"))
    assert r.status_code == 200
    v = r.json()
    assert v["decision"] == "ACCEPT"
    assert v["similarity"] > settings.t_sim
    assert v["thresholds"] == {"t_sim": settings.t_sim, "t_fake": settings.t_fake}
    assert v["face_box"] is not None

    v = _verify(client, iid, img("FG:PERSON=mallory;")).json()
    assert v["decision"] == "REJECT_MISMATCH" and v["similarity"] < settings.t_sim

    v = _verify(client, iid, img("FG:PERSON=alice;", "FG:FAKE")).json()
    assert v["decision"] == "REJECT_FAKE" and v["fake_probability"] >= settings.t_fake

    v = _verify(client, iid, img("FG:NOFACE")).json()
    assert v["decision"] == "NO_FACE"
    assert v["similarity"] is None and v["face_box"] is None
    assert v["thresholds"]["t_sim"] == settings.t_sim

    assert client.delete(f"/api/identities/{iid}").status_code == 204
    assert client.get("/api/identities").json() == []
    assert client.delete(f"/api/identities/{iid}").status_code == 404
    assert _verify(client, iid, img("FG:PERSON=alice;")).status_code == 404


def test_enroll_partial_rejections(client):
    r = client.post(
        "/api/enroll",
        data={"name": "bob"},
        files=_files(
            img("FG:PERSON=bob;"),
            img("FG:NOFACE"),
            b"garbage bytes",
            img("FG:PERSON=bob;", "FG:FAKE"),
        ),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["images_used"] == 1
    reasons = {x["index"]: x["reason"] for x in body["rejected"]}
    assert reasons[1] == "no face detected"
    assert reasons[2] == "invalid image"
    assert reasons[3] == "flagged as fake (p=0.90)"


def test_enroll_all_rejected_400(client):
    r = client.post("/api/enroll", data={"name": "eve"}, files=_files(img("FG:NOFACE")))
    assert r.status_code == 400
    assert r.json()["detail"]["rejected"] == [{"index": 0, "reason": "no face detected"}]
    assert client.get("/api/identities").json() == []


def test_enroll_duplicate_409(client):
    assert _enroll(client).status_code == 200
    assert _enroll(client).status_code == 409


def test_enroll_file_count_400(client):
    assert client.post("/api/enroll", data={"name": "x"}).status_code == 400
    n = settings.enroll_max_images + 1
    r = client.post("/api/enroll", data={"name": "x"}, files=_files(*[img("FG:PERSON=x;")] * n))
    assert r.status_code == 400


def test_verify_unknown_identity_404(client):
    assert _verify(client, 9999, img("FG:PERSON=alice;")).status_code == 404


def test_verify_garbage_400(client):
    iid = _enroll(client).json()["identity_id"]
    assert _verify(client, iid, b"not an image at all").status_code == 400


def test_detect(client):
    def detect(blob):
        return client.post("/api/detect", files={"file": ("x.png", blob, "image/png")})

    d = detect(img("FG:PERSON=a;")).json()
    assert d["fake_probability"] < settings.t_fake and d["label"] == "real" and d["face_box"]

    d = detect(img("FG:FAKE")).json()
    assert d["fake_probability"] >= settings.t_fake and d["label"] == "fake"

    r = detect(img("FG:NOFACE"))
    assert r.status_code == 200
    assert r.json() == {
        "fake_probability": None, "label": None, "face_box": None,
        "warnings": ["no face detected"],
    }

    assert detect(b"garbage").status_code == 400


def test_real_jpeg_same_file_accepts(client):
    """Without markers the fake keys off the file hash: same photo ACCEPT, other photo MISMATCH."""
    photo = b"\xff\xd8\xff\xe0" + b"photo-one"
    r = client.post("/api/enroll", data={"name": "zed"}, files=_files(photo))
    iid = r.json()["identity_id"]
    assert _verify(client, iid, photo).json()["decision"] == "ACCEPT"
    other = b"\xff\xd8\xff\xe0" + b"photo-two"
    assert _verify(client, iid, other).json()["decision"] == "REJECT_MISMATCH"
