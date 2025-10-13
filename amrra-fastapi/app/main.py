import os, json, subprocess, pathlib, time, asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from app.core.experiment import run_toy_experiment
from app.core.hashcanon import canonical_string, sha256_hex
from app.core.pdfgen import render_pdf
import re
import urllib.request, base64, copy
from app.core.retrieval import build_retriever
from app.core.ml_experiment import run_ml_experiment
from app.core.hypothesis import generate_hypotheses
from app.core.websearch import WebSearch
from app.core.llm import (summarize_sources_with_llm, 
                          build_conclusion_with_llm, 
                          build_hypotheses_with_llm,
                          build_study_design_with_llm
                          )
load_dotenv()

ARTIFACT_DIR = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

PORT = int(os.getenv("PORT", "8000"))
HEDERA_NETWORK = os.getenv("HEDERA_NETWORK", "testnet")
OP_ID = os.getenv("HEDERA_OPERATOR_ID")
OP_KEY = os.getenv("HEDERA_OPERATOR_KEY")
TOPIC_ID = os.getenv("HCS_TOPIC_ID")
CORPUS_DIR = os.getenv("CORPUS_DIR")
RETRIEVER = None
app = FastAPI(title="AMRRA Research Lab (FastAPI)")

def _valid_topic_id(x: str) -> bool:
    return bool(re.fullmatch(r"0\.0\.\d+", (x or "").strip()))

class RunReq(BaseModel):
    question: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.on_event("startup")
def _init_retriever():
    global RETRIEVER
    try:
        RETRIEVER = build_retriever(CORPUS_DIR)
        print(f"[retrieval] index ready ({len(RETRIEVER.docs)} docs)")
    except Exception as e:
        print("[retrieval] failed to init:", e)

WEB: WebSearch | None = None
@app.on_event("startup")
def _init_web():
    global WEB
    try:
        WEB = WebSearch.from_env()
        print(f"[web] provider={WEB.provider}")
    except Exception as e:
        print("[web] init failed:", e)

#llm test
@app.get("/api/llm/hypotheses")
def llm_hypotheses(q: str = "Will RF outperform LR on AUC?"):
    data, meta = build_hypotheses_with_llm(q, {"name":"sklearn_breast_cancer","target_name":"malignant_vs_benign"}, [])
    return {"meta": meta, "data": data}

@app.get("/api/retrieval/web")
def web_retrieval(q: str, k: int = 3):
    if not WEB:
        raise HTTPException(500, "web search not initialized")
    results = asyncio.run(WEB.search_and_summarize(q, k))
    return {"query": q, "results": results}

@app.get("/api/retrieval/search")
def retrieval_search(q: str, k: int = 3):
    if not RETRIEVER:
        raise HTTPException(500, "retriever not initialized")
    return {"query": q, "results": RETRIEVER.search(q, top_k=k)}

@app.post("/api/run")
def run(req: RunReq):
    run_id = f"run_{int(time.time()*1000)}"
    json_name = f"{run_id}.json"
    pdf_name  = f"{run_id}.pdf"
    json_path = ARTIFACT_DIR / json_name
    pdf_path  = ARTIFACT_DIR / pdf_name

    question = (req.question or "")[:500]
    mode = str(getattr(req, "experiment", None) or "ttest").lower().strip()

    # ---- retrieval helpers (web + local, dedup, cap) ----
    def _dedupe_sources(items, max_k=5):
        seen = set()
        out = []
        for s in items:
            key = (s.get("url")
                   or (s.get("source") or {}).get("name")
                   or s.get("title"))
            if not key:
                continue
            k = key.strip().lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
            if len(out) >= max_k:
                break
        return out

    def _get_sources(q: str, fallback_q: str, top_k_web=3, top_k_local=3, final_cap=5):
        use_web = os.getenv("ENABLE_WEB_RETRIEVAL", "true").lower() == "true"
        q_eff = (q or fallback_q).strip() or fallback_q

        web_sources = []
        if use_web and WEB:
            try:
                web_sources = asyncio.run(WEB.search_and_summarize(q_eff, k=top_k_web))
            except Exception:
                web_sources = []

        local_sources = []
        if RETRIEVER:
            try:
                local_sources = RETRIEVER.search(q_eff, top_k=top_k_local)
            except Exception:
                local_sources = []

        # order: web first (fresh), then local (context)
        merged = (web_sources or []) + (local_sources or [])
        return _dedupe_sources(merged, max_k=final_cap)

    # ---------- Build report (ML or t-test), with FINAL artifacts (names only) ----------
    if mode == "ml":
        ml = run_ml_experiment()
        ds_meta = {"name": ml["dataset"]["name"], "target_name": ml["dataset"]["target_name"]}
        sources = _get_sources(question, "model performance comparison")
        summaries = []
        summarizer_id = None
        if os.getenv("ENABLE_LLM", "false").lower() == "true" and sources:
            try:
                sdata, smeta = summarize_sources_with_llm(sources, max_items=3)
                summaries = sdata.get("summaries", [])
                summarizer_id = f"{smeta.get('provider')}:{smeta.get('model')}"
            except Exception:
                summaries = []
                summarizer_id = None
        hyps = generate_hypotheses(question, ds_meta, sources)
        conclusion = {}
        concluder_id = None
        if os.getenv("ENABLE_LLM","false").lower() == "true":
            try:
                plan_for_llm = {
                    "design": ml["design"]["task"],
                    "primary_metric": ml["design"]["primary_metric"],
                    "test_size": ml["design"]["test_size"],
                    "seed": ml["design"]["seed"]
                }
                results_for_llm = {
                    "dataset": ml["dataset"],
                    "metrics": ml["metrics"],
                    "comparison": ml["comparison"]
                }
                cdata, cmeta = build_conclusion_with_llm(question, plan_for_llm, results_for_llm, sources, hyps)
                conclusion = {
                    "text": cdata.get("conclusion"),
                    "strength": cdata.get("strength_of_evidence"),
                    "limitations": cdata.get("limitations"),
                    "next_steps": cdata.get("next_steps")
                }
                concluder_id = f"{cmeta.get('provider')}:{cmeta.get('model')}"
            except Exception:
                conclusion = {}
                concluder_id = None

        study_rec = {}
        recommender_id = None
        if os.getenv("ENABLE_LLM","false").lower() == "true":
            try:
                plan_for_llm = {
                    "design": ml["design"]["task"],
                    "primary_metric": ml["design"]["primary_metric"],
                    "test_size": ml["design"]["test_size"],
                    "seed": ml["design"]["seed"]
                }
                results_for_llm = {
                    "dataset": ml["dataset"],
                    "metrics": ml["metrics"],
                    "comparison": ml["comparison"]
                }
                sdata, smeta = build_study_design_with_llm(
                    question=question,
                    plan=plan_for_llm,
                    results=results_for_llm,
                    sources=sources,
                    constraints={"budget": None, "compute": "CPU", "deadline_days": 7}
                )
                study_rec = sdata
                recommender_id = f"{smeta.get('provider')}:{smeta.get('model')}"
            except Exception:
                study_rec = {}
                recommender_id = None

        env_models = {"generator": "template-hypotheses-v1", "eval": "sklearn-1.5.1"}
        if summarizer_id:
            env_models["summarizer"] = summarizer_id
        if concluder_id:
            env_models["concluder"] = concluder_id
        if recommender_id:
            env_models["recommender"] = recommender_id
        
        

        report = {
            "schema": "amrra.report.v1",
            "run_id": run_id,
            "request": {"question": question},
            "retrieval": {"sources": sources},
            "hypotheses": hyps,
            "plan": {
                "design": ml["design"]["task"],
                "primary_metric": ml["design"]["primary_metric"],
                "test_size": ml["design"]["test_size"],
                "seed": ml["design"]["seed"]
            },
            "results": {
                "dataset": ml["dataset"],
                "metrics": ml["metrics"],
                "comparison": ml["comparison"]
            },
            "conclusion": conclusion,
            "plan_recommendations": study_rec,
            "environment": {
                "datasets": [{"name": ml["dataset"]["name"], "hash": "sha256:sklearn-canonical"}],
                "code": {"image": "local-demo"},
                "models": env_models,
                "hardware": "cpu"
            },
            "artifacts": {
                "json": {"type": "local", "name": json_name},
                "pdf":  {"type": "local", "name": pdf_name}
            },
            "timestamp_unix": int(time.time()),
            "report_hash": None
        }
    else:
        results = run_toy_experiment(12345)
        sources = _get_sources(question, "two-sample t-test effect size")
        summaries = []
        summarizer_id = None
        if os.getenv("ENABLE_LLM", "false").lower() == "true" and sources:
            try:
                sdata, smeta = summarize_sources_with_llm(sources, max_items=3)
                summaries = sdata.get("summaries", [])
                summarizer_id = f"{smeta.get('provider')}:{smeta.get('model')}"
            except Exception:
                summaries = []
                summarizer_id = None
        hyps = generate_hypotheses(question, {"name":"toy","target_name":"group"}, sources)
        conclusion = {}
        concluder_id = None
        if os.getenv("ENABLE_LLM","false").lower() == "true":
            try:
                plan_for_llm = {
                    "design": results["design"],
                    "alpha": results["alpha"],
                    "seed": 12345,
                    "n": results["n"]
                }
                results_for_llm = {
                    "p_value": results["p"],
                    "effect_size": results["effect_size"],
                    "ci": results["ci"]
                }
                hyps_tt = ["H0: no difference", "H1: group B differs"]
                cdata, cmeta = build_conclusion_with_llm(question, plan_for_llm, results_for_llm, sources, hyps_tt)
                conclusion = {
                    "text": cdata.get("conclusion"),
                    "strength": cdata.get("strength_of_evidence"),
                    "limitations": cdata.get("limitations"),
                    "next_steps": cdata.get("next_steps")
                }
                concluder_id = f"{cmeta.get('provider')}:{cmeta.get('model')}"
            except Exception:
                conclusion = {}
                concluder_id = None

        study_rec = {}
        recommender_id = None
        if os.getenv("ENABLE_LLM","false").lower() == "true":
            try:
                plan_for_llm = {
                    "design": results["design"],
                    "alpha": results["alpha"],
                    "seed": 12345,
                    "n": results["n"]
                }
                results_for_llm = {
                    "p_value": results["p"],
                    "effect_size": results["effect_size"],
                    "ci": results["ci"]
                }
                sdata, smeta = build_study_design_with_llm(
                    question=question,
                    plan=plan_for_llm,
                    results=results_for_llm,
                    sources=sources,
                    constraints={"budget": None, "compute": "CPU", "deadline_days": 7}
                )
                study_rec = sdata
                recommender_id = f"{smeta.get('provider')}:{smeta.get('model')}"
            except Exception:
                study_rec = {}
                recommender_id = None

        env_models = {"generator": "template-hypotheses-v1", "eval": "sklearn-1.5.1"}
        if summarizer_id:
            env_models["summarizer"] = summarizer_id
        if concluder_id:
            env_models["concluder"] = concluder_id
        if recommender_id:
            env_models["recommender"] = recommender_id

        report = {
            "schema": "amrra.report.v1",
            "run_id": run_id,
            "request": {"question": question},
            "retrieval": {"sources": sources},
            "hypotheses": ["H0: no difference", "H1: group B differs"],
            "plan": {"design": results["design"], "n": results["n"], "alpha": results["alpha"], "seed": 12345},
            "results": {"p": results["p"], "effect_size": results["effect_size"], "ci": results["ci"]},
            "conclusion": conclusion,
            "plan_recommendations": study_rec,
            "environment": {
                "datasets": [{"name": "toy", "hash": "sha256:static-seeded"}],
                "code": {"image": "local-demo"},
                "models": env_models,
                "hardware": "cpu"
            },
            "artifacts": {
                "json": {"type": "local", "name": json_name},
                "pdf":  {"type": "local", "name": pdf_name}
            },
            "timestamp_unix": int(time.time()),
            "report_hash": None
        }

    # ---------- Canonical hash AFTER artifacts are set; BEFORE any writes ----------
    can = canonical_string(report)             # omits 'report_hash' by design
    h = sha256_hex(can)
    report["report_hash"] = f"sha256:{h}"

    # ---------- Write JSON exactly once ----------
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---------- Sanity re-read & re-hash to guarantee consistency ----------
    with open(json_path, "r", encoding="utf-8") as f:
        _disk = json.load(f)
    _disk_can = canonical_string(_disk)
    _disk_hash = "sha256:" + sha256_hex(_disk_can)
    if _disk_hash != report["report_hash"]:
        raise RuntimeError(f"Post-write rehash mismatch: disk={_disk_hash} computed={report['report_hash']}")

    # ---------- Render PDF once (MUST NOT mutate 'report') ----------
    render_pdf(report, str(pdf_path))

    # ---------- HCS anchor via Node (names only; no absolute paths) ----------
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
                "artifacts": report["artifacts"],
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
                timeout=45
            ).decode("utf-8").strip()
            hcs = json.loads(out)
        except subprocess.CalledProcessError as e:
            hcs = {"error": e.output.decode("utf-8", errors="ignore")}
        except Exception as e:
            hcs = {"error": str(e)}

    return {
        "ok": True,
        "run_id": run_id,
        "report_hash": report["report_hash"],
        "artifacts": {
            "json": f"/api/artifacts/{json_name}",
            "pdf":  f"/api/artifacts/{pdf_name}"
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
    return "sha256:" + sha256_hex(canonical_string(report))

def _hash_legacy_empty_artifacts(report: dict) -> str:
    r = json.loads(json.dumps(report))  # deep copy
    r["artifacts"] = {}
    return "sha256:" + sha256_hex(canonical_string(r))

def _hash_sortkeys_include_field(report: dict) -> str:
    """Fallback: some old code might have hashed with sort_keys and DIDN'T drop 'report_hash'."""
    r = copy.deepcopy(report)
    s = json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + sha256_hex(s)

def _hash_sortkeys_drop_field(report: dict) -> str:
    """Fallback: sort_keys but DO drop 'report_hash' (approx to older naive canon)."""
    r = copy.deepcopy(report)
    r.pop("report_hash", None)
    s = json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + sha256_hex(s)

@app.get("/api/hcs/verify/{run_id}")
def verify_run(run_id: str):
    # 1) load local report
    json_path = (ARTIFACT_DIR / f"{run_id}.json").resolve()
    if not str(json_path).startswith(str(ARTIFACT_DIR.resolve())) or not json_path.exists():
        raise HTTPException(404, f"local report not found: {json_path.name}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception as e:
        raise HTTPException(500, f"failed reading local report: {e}")

    file_embedded_hash = report.get("report_hash")

    # 2) fetch on-chain record
    if not TOPIC_ID:
        raise HTTPException(400, "HCS_TOPIC_ID not set")

    url = f"https://testnet.mirrornode.hedera.com/api/v1/topics/{TOPIC_ID}/messages?limit=100"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
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
            "on_chain": None,
            "verified": False,
            "reason": "Run not found on mirror (yet). Try again shortly."
        }

    on_chain_hash = mirror["report_hash"]

    # 3) compute candidate hashes
    candidates = {
        "current_canonical": _hash_current(report),
        "legacy_empty_artifacts": _hash_legacy_empty_artifacts(report),
        "sortkeys_include_report_hash": _hash_sortkeys_include_field(report),
        "sortkeys_drop_report_hash": _hash_sortkeys_drop_field(report),
        "file_embedded_report_hash": file_embedded_hash or "<none>"
    }

    mode = None
    verified = False
    for name, val in candidates.items():
        if val == on_chain_hash:
            mode = name
            verified = True
            break

    return {
        "run_id": run_id,
        "on_chain": {
            "sequence_number": mirror["sequence_number"],
            "consensus_timestamp": mirror["consensus_timestamp"],
            "report_hash": on_chain_hash
        },
        "candidates": candidates,
        "verified": verified,
        "mode": mode
    }
