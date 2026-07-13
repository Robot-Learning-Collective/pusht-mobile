"""PushT episode collector.

Stores each successful episode as one JSON file in server/episodes/.
The filename encodes submission time and participant name
(<epoch_ns>_<name-slug>.json), so stats need only a directory listing.
"""

import json
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

# On a hosted VM point this at a persistent volume (e.g. /workspace/episodes).
EPISODES_DIR = Path(os.environ.get("PUSHT_EPISODES_DIR", Path(__file__).parent / "episodes"))
EPISODES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "anon"


@app.post("/api/episodes")
async def save_episode(req: Request):
    try:
        ep = await req.json()
    except Exception:
        raise HTTPException(400, "invalid JSON")

    name = str(ep.get("name", "")).strip()
    frames = ep.get("frames")
    if not name:
        raise HTTPException(422, "name is required")
    if not ep.get("success"):
        raise HTTPException(422, "only successful episodes are accepted")
    if not isinstance(frames, list) or len(frames) < 2:
        raise HTTPException(422, "episode has no frames")

    fname = f"{time.time_ns()}_{slugify(name)}.json"
    # Write to a temp name, then rename: a killed request never leaves a
    # half-written .json behind.
    tmp = EPISODES_DIR / (fname + ".part")
    tmp.write_text(json.dumps(ep))
    os.rename(tmp, EPISODES_DIR / fname)

    total = sum(1 for _ in EPISODES_DIR.glob("*.json"))
    return {"file": fname, "total": total}


@app.get("/api/stats")
def stats():
    counts: dict[str, int] = {}
    for f in EPISODES_DIR.glob("*.json"):
        slug = f.stem.split("_", 1)[1] if "_" in f.stem else "anon"
        counts[slug] = counts.get(slug, 0) + 1
    by_name = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "total": sum(counts.values()),
        "byName": [{"name": n, "count": c} for n, c in by_name],
    }


# Production: serve the built frontend (npm run build:wasm && npm run build)
# from this same process, so phones talk to a single origin — no proxy needed.
# Mounted last so /api/* routes above take precedence. In dev, Vite serves the
# frontend instead and proxies /api here.
DIST = Path(__file__).parent.parent / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
