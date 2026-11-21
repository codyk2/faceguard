"""Enroll N well-known LFW people through the running API.

Usage (from backend/, with the API running on :8000):
    uv run python ../scripts/seed_enroll.py                     # 10 people, 4 images each
    uv run python ../scripts/seed_enroll.py --n 5 --per-person 3 --api http://localhost:8000
    uv run python ../scripts/seed_enroll.py --dry-run           # only write the manifest

For each person the first `--per-person` LFW images (sorted, _0001 first) are enrolled;
the rest are held out as probes. data/seed_manifest.json records both lists, plus the
identity_id returned by the API. People already enrolled (same display name) are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import _common as C
import httpx

DEFAULT_PEOPLE = [
    "George_W_Bush", "Colin_Powell", "Tony_Blair", "Donald_Rumsfeld", "Gerhard_Schroeder",
    "Ariel_Sharon", "Hugo_Chavez", "Junichiro_Koizumi", "Jean_Chretien", "John_Ashcroft",
    # spares, used if --n > 10 or one of the above is missing
    "Jacques_Chirac", "Serena_Williams", "Vladimir_Putin", "Luiz_Inacio_Lula_da_Silva",
    "Gloria_Macapagal_Arroyo", "Jennifer_Capriati", "Lleyton_Hewitt", "Laura_Bush",
]
MIN_IMAGES = 8


def display_name(lfw_name: str) -> str:
    return lfw_name.replace("_", " ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8000", help="API base URL (no /api suffix)")
    ap.add_argument("--n", type=int, default=10, help="number of people")
    ap.add_argument("--per-person", type=int, default=4, help="images enrolled per person")
    ap.add_argument("--dry-run", action="store_true", help="pick people and write manifest, no HTTP")
    args = ap.parse_args()

    people = C.lfw_people(min_images=MIN_IMAGES)
    if not people:
        print(f"no LFW data under {C.LFW_DIR}; run download_data.py --only lfw first", file=sys.stderr)
        return 2
    chosen = [p for p in DEFAULT_PEOPLE if p in people][: args.n]
    if len(chosen) < args.n:  # top up with the people who have the most images
        extra = sorted((p for p in people if p not in chosen), key=lambda p: -len(people[p]))
        chosen += extra[: args.n - len(chosen)]

    base = args.api.rstrip("/") + "/api"
    existing: dict[str, int] = {}
    client = None
    if not args.dry_run:
        client = httpx.Client(timeout=120.0)
        try:
            r = client.get(f"{base}/identities")
            r.raise_for_status()
        except httpx.HTTPError as e:
            print(f"cannot reach API at {base}: {e}", file=sys.stderr)
            return 2
        existing = {i["name"]: i["identity_id"] for i in r.json()}

    # keep earlier manifest entries (e.g. identity ids from a previous run)
    old = {}
    if C.SEED_MANIFEST.exists():
        try:
            old = {e["lfw_name"]: e for e in json.loads(C.SEED_MANIFEST.read_text())["people"]}
        except Exception:
            old = {}

    entries = []
    ok = True
    for lfw_name in chosen:
        imgs = people[lfw_name]
        enroll, probe = imgs[: args.per_person], imgs[args.per_person:]
        name = display_name(lfw_name)
        entry = {
            "lfw_name": lfw_name,
            "name": name,
            "identity_id": existing.get(name, old.get(lfw_name, {}).get("identity_id")),
            "enroll_images": [str(p.relative_to(C.DATA_DIR)) for p in enroll],
            "probe_images": [str(p.relative_to(C.DATA_DIR)) for p in probe],
            "status": "planned",
            "rejected": [],
        }
        if args.dry_run:
            pass
        elif name in existing:
            entry["status"] = "already_enrolled"
            print(f"  skip  {name:28s} (identity {existing[name]})")
        else:
            files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in enroll]
            r = client.post(f"{base}/enroll", data={"name": name}, files=files)
            if r.status_code == 200:
                body = r.json()
                entry.update(identity_id=body["identity_id"], status="enrolled",
                             images_used=body["images_used"], rejected=body["rejected"])
                rej = f", rejected {body['rejected']}" if body["rejected"] else ""
                print(f"  enrol {name:28s} -> id {body['identity_id']}, "
                      f"{body['images_used']}/{len(enroll)} used{rej}")
            elif r.status_code == 409:
                entry["status"] = "already_enrolled"
                print(f"  skip  {name:28s} (409 already exists)")
            else:
                ok = False
                entry["status"] = f"error {r.status_code}"
                entry["error"] = r.text[:500]
                print(f"  FAIL  {name:28s} HTTP {r.status_code}: {r.text[:200]}")
        entries.append(entry)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "api": None if args.dry_run else base,
        "per_person": args.per_person,
        "paths_relative_to": "data/",
        "people": entries,
    }
    C.SEED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    C.SEED_MANIFEST.write_text(json.dumps(manifest, indent=2))
    n_probe = sum(len(e["probe_images"]) for e in entries)
    print(f"wrote {C.SEED_MANIFEST} ({len(entries)} people, {n_probe} held-out probe images)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
