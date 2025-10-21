# app/api.py
import os, json, pathlib, time, re
from typing import Optional, Literal, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rq import Queue
from redis import Redis
from rq.job import Job

from app.core.hashcanon import canonical_string, sha256_hex

# dirs
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
RUNS_DIR = BASE_DIR / "runs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CORPUS_DIR = os.getenv("CORPUS_DIR", str(BASE_DIR / "corpus"))

# RQ queue
redis = Redis.from_url(REDIS_URL)
q = Queue("amrra", connection=redis)

app = FastAPI(title="AMRRA Back End")

class RunReq(BaseModel):
    question: str = Field(..., max_length=500)
    experiment: Optional[Literal["ml", "ttest", "literature"]] = None
    class Config:
        extra = "forbid"

def _run_manifest_path(run_id: str) -> pathlib.Path:
    return RUNS_DIR / f"{run_id}.json"

@app.get("/health")
def health():
    try:
        redis.ping()
        ok = True
    except Exception:
        ok = False
    return {"ok": ok}

@app.post("/api/run")
def enqueue_run(req: RunReq):
    run_id = f"run_{int(time.time()*1000)}"
    # store a manifest immediately
    manifest = {
        "run_id": run_id,
        "request": {"question": req.question, "experiment": req.experiment},
        "status": "queued",
        "artifacts": {},
        "ts": int(time.time())
    }
    _run_manifest_path(run_id).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # enqueue
    job: Job = q.enqueue("worker.tasks.run_pipeline",
        run_id, req.question, req.experiment,
        job_timeout=60*15, result_ttl=60*60*24, ttl=60*60*24)

    return {"ok": True, "run_id": run_id, "job_id": job.get_id(), "status": "queued"}

@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    manifest_path = _run_manifest_path(run_id)
    if not manifest_path.exists():
        raise HTTPException(404, "run not found")
    data = json.loads(manifest_path.read_text("utf-8"))
    return data

@app.get("/api/artifacts/{filename}")
def artifacts(filename: str):
    p = (ARTIFACT_DIR / filename).resolve()
    if not str(p).startswith(str(ARTIFACT_DIR.resolve())):
        raise HTTPException(400, "bad path")
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(p))