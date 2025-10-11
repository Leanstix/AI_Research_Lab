import os, json, subprocess, pathlib, time
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from app.core.experiment import run_toy_experiment
from app.core.hashcanon import canonical_string, sha256_hex
from app.core.pdfgen import render_pdf
import re
import urllib.request, base64
load_dotenv()

ARTIFACT_DIR = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

PORT = int(os.getenv("PORT", "8000"))
HEDERA_NETWORK = os.getenv("HEDERA_NETWORK", "testnet")
OP_ID = os.getenv("HEDERA_OPERATOR_ID")
OP_KEY = os.getenv("HEDERA_OPERATOR_KEY")
TOPIC_ID = os.getenv("HCS_TOPIC_ID")  # may be empty

app = FastAPI(title="AMRRA Research Lab (FastAPI)")

def _valid_topic_id(x: str) -> bool:
    return bool(re.fullmatch(r"0\.0\.\d+", (x or "").strip()))

class RunReq(BaseModel):
    question: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/run")
def run(req: RunReq):
    run_id = f"run_{int(time.time()*1000)}"
    json_name = f"{run_id}.json"
    pdf_name  = f"{run_id}.pdf"
    json_path = ARTIFACT_DIR / json_name
    pdf_path  = ARTIFACT_DIR / pdf_name
    question = (req.question or "")[:500]

    # 1) experiment
    results = run_toy_experiment(12345)

    # 2) build report (report_hash will be filled after hashing canonical JSON)
    report = {
        "schema": "amrra.report.v1",
        "run_id": run_id,
        "request": {"question": (req.question or "")[:500]},
        "retrieval": {"sources": []},
        "hypotheses": ["H0: no difference", "H1: group B differs"],
        "plan": {"design": results["design"], "n": results["n"], "alpha": results["alpha"], "seed": 12345},
        "results": {"p": results["p"], "effect_size": results["effect_size"], "ci": results["ci"]},
        "environment": {
            "datasets": [{"name": "toy", "hash": "sha256:static-seeded"}],
            "code": {"image": "local-demo"},
            "models": {"generator": "prompt-template-v1", "eval": "custom-py"},
            "hardware": "cpu"
        },
        "artifacts": { "json": {"type": "local", "name": json_name},
                    "pdf":  {"type": "local", "name": pdf_name} },
        "timestamp_unix": int(time.time()),
        "report_hash": None
    }

    # 3) canonical hash (omit report_hash during hashing)
    can = canonical_string(report)
    h = sha256_hex(can)
    report["report_hash"] = f"sha256:{h}"

    # 4) write JSON
    json_path = ARTIFACT_DIR / f"{run_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 5) render PDF
    render_pdf(report, str(pdf_path))
    # include json path so it shows in pdf
    rpt_for_pdf = dict(report)
    rpt_for_pdf["artifacts"] = {"json_path": str(json_path)}
    render_pdf(rpt_for_pdf, str(pdf_path))

    # update json with artifacts
    report["artifacts"] = {"json_path": str(json_path), "pdf_path": str(pdf_path)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 6) HCS anchor via Node (optional if TOPIC_ID empty)
    hcs = None
    
    if TOPIC_ID and OP_ID and OP_KEY:
        if not _valid_topic_id(TOPIC_ID):
            raise HTTPException(400, "bad HCS_TOPIC_ID")
        payload = {
            "topicId": TOPIC_ID,
            "message": {
                "schema": "amrra.hcs.v1",
                "run_id": run_id,
                "report_hash": report["report_hash"],
                "artifacts": report["artifacts"],   # just names/types
                "models": report["environment"]["models"],
                "datasets": report["environment"]["datasets"],
                "credits_spent": 5,
                "signer": OP_ID,
                "ts": report["timestamp_unix"]
            },
            "network": HEDERA_NETWORK,
            "operatorId": OP_ID,
            "operatorKey": OP_KEY,
            "awaitReceipt": False
        }
        node_script = (pathlib.Path(__file__).resolve().parent / "integrations" / "hcs_client.js")
        try:
            out = subprocess.check_output(
                ["node", str(node_script), json.dumps(payload)],
                stderr=subprocess.STDOUT,
                timeout=45             # <- shorter, because we return fast now
            ).decode("utf-8").strip()
            hcs = json.loads(out)
        except subprocess.CalledProcessError as e:
            # Don’t fail the whole run if Hedera is down; return the error so you can see it
            hcs = {"error": e.output.decode("utf-8", errors="ignore")}
        except Exception as e:
            hcs = {"error": str(e)}

    return {
        "ok": True,
        "run_id": run_id,
        "report_hash": report["report_hash"],
        "artifacts": {
            "json": f"/api/artifacts/{run_id}.json",
            "pdf": f"/api/artifacts/{run_id}.pdf"
        },
        "hcs": hcs
    }

@app.get("/api/artifacts/{filename}")
def artifacts(filename: str):
    p = (ARTIFACT_DIR / filename).resolve()
    if not str(p).startswith(str(ARTIFACT_DIR.resolve())):
        raise HTTPException(400, "bad path")
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(p))

@app.get("/api/hcs/recent")
def hcs_recent(limit: int = 20):
    if not TOPIC_ID:
        raise HTTPException(400, "HCS_TOPIC_ID not set")
    if limit < 1 or limit > 100:
        raise HTTPException(400, "limit out of range (1..100)")

    url = f"https://testnet.mirrornode.hedera.com/api/v1/topics/{TOPIC_ID}/messages?limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise HTTPException(502, f"mirror fetch failed: {e}")

    out = []
    for m in data.get("messages", []):
        raw = m.get("message", "")
        decoded = None
        try:
            decoded = json.loads(base64.b64decode(raw).decode("utf-8"))
        except Exception:
            # leave decoded=None if not JSON
            pass
        out.append({
            "sequence_number": int(m.get("sequence_number", 0)),
            "consensus_timestamp": m.get("consensus_timestamp"),
            "payer_account_id": m.get("payer_account_id"),
            "message_decoded": decoded,     # your JSON payload if any
        })

    return {"topic_id": TOPIC_ID, "messages": out}

def _hash_current(report: dict) -> str:
    # Hash the report exactly as it is now (artifacts included), omitting report_hash via canonical_string()
    return "sha256:" + sha256_hex(canonical_string(report))

def _hash_legacy(report: dict) -> str:
    # Emulate old behavior: artifacts were empty when the hash was computed
    r = json.loads(json.dumps(report))   # deep copy
    r["artifacts"] = {}
    return "sha256:" + sha256_hex(canonical_string(r))

@app.get("/api/hcs/verify/{run_id}")
def verify_run(run_id: str):
    # 1) load local JSON
    json_path = (ARTIFACT_DIR / f"{run_id}.json").resolve()
    if not str(json_path).startswith(str(ARTIFACT_DIR.resolve())) or not json_path.exists():
        raise HTTPException(404, f"local report not found: {json_path.name}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception as e:
        raise HTTPException(500, f"failed reading local report: {e}")

    # Compute both hashes
    local_hash_cur = _hash_current(report)
    local_hash_legacy = _hash_legacy(report)

    # 2) fetch recent mirror msgs and find this run_id
    if not TOPIC_ID:
        raise HTTPException(400, "HCS_TOPIC_ID not set")

    url = f"https://testnet.mirrornode.hedera.com/api/v1/topics/{TOPIC_ID}/messages?limit=50"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise HTTPException(502, f"mirror fetch failed: {e}")

    mirror = None
    for m in data.get("messages", []):
        try:
            decoded = json.loads(base64.b64decode(m.get("message","")).decode("utf-8"))
        except Exception:
            continue
        if isinstance(decoded, dict) and decoded.get("run_id") == run_id:
            mirror = {
                "sequence_number": int(m.get("sequence_number", 0)),
                "consensus_timestamp": m.get("consensus_timestamp"),
                "report_hash": decoded.get("report_hash"),
                "payload": decoded,
            }
            break

    if not mirror:
        return {
            "run_id": run_id,
            "local_hash": local_hash_cur,
            "local_hash_legacy": local_hash_legacy,
            "on_chain": None,
            "verified": False,
            "mode": None,
            "reason": "Run not found on mirror (yet). Try again in a few seconds.",
        }

    on_chain_hash = mirror["report_hash"]
    mode = None
    verified = False

    if on_chain_hash == local_hash_cur:
        verified = True
        mode = "current"
    elif on_chain_hash == local_hash_legacy:
        verified = True
        mode = "legacy"

    return {
        "run_id": run_id,
        "local_hash": local_hash_cur,          # current scheme hash
        "local_hash_legacy": local_hash_legacy,
        "on_chain": {
            "sequence_number": mirror["sequence_number"],
            "consensus_timestamp": mirror["consensus_timestamp"],
            "report_hash": on_chain_hash,
        },
        "verified": bool(verified),
        "mode": mode,
    }
