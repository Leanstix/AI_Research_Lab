import os, json, pathlib, time, subprocess, urllib.request, base64, re, asyncio
from typing import Optional, Literal, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.core.hashcanon import canonical_string, sha256_hex
from app.core.pdfgen import render_pdf
from app.core.retrieval import build_retriever
from app.core.ml_experiment import run_ml_experiment
from app.core.hypothesis import generate_hypotheses
from app.core.websearch import WebSearch
from app.core.llm import (
    summarize_sources_with_llm,
    build_conclusion_with_llm,
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

if CORPUS_DIR:
    pathlib.Path(CORPUS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"[retrieval] corpus dir: {pathlib.Path(CORPUS_DIR).resolve()}")

RETRIEVER = None
WEB: WebSearch | None = None

app = FastAPI(title="AMRRA Research Lab (FastAPI)")

def _valid_topic_id(x: str) -> bool:
    return bool(re.fullmatch(r"0\.0\.\d+", (x or "").strip()))

class RunReq(BaseModel):
    question: str = Field(..., max_length=500)
    # Optional override; if omitted we auto-pick based on question semantics
    experiment: Optional[Literal["ml", "ttest", "literature"]] = None
    class Config:
        extra = "forbid"

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

@app.on_event("startup")
def _init_web():
    global WEB
    try:
        WEB = WebSearch.from_env()
        print(f"[web] provider={WEB.provider}")
    except Exception as e:
        print("[web] init failed:", e)

# ---------- helpers ----------
def _dedupe_sources(items: List[Dict[str,Any]], max_k=5) -> List[Dict[str,Any]]:
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

def _get_sources(q: str, top_k_web=3, top_k_local=3, final_cap=5) -> List[Dict[str,Any]]:
    """
    Dynamic retrieval: try local TF-IDF; if OOV/empty it hydrates from web and persists into CORPUS_DIR.
    Also fetch a few fresh web docs and merge/dedupe.
    """
    q_eff = (q or "").strip() or "baseline model comparison performance"

    web_sources = []
    if WEB:
        try:
            web_sources = asyncio.run(WEB.search_and_summarize(q_eff, k=top_k_web))
        except Exception:
            web_sources = []

    local_sources = []
    if RETRIEVER:
        try:
            # dynamic: will hydrate from web if empty / OOV and persist into CORPUS_DIR
            local_sources = RETRIEVER.search(q_eff, top_k=top_k_local, web=WEB, web_k=final_cap, hydrate=True)
        except Exception:
            local_sources = []

    merged = (web_sources or []) + (local_sources or [])
    return _dedupe_sources(merged, max_k=final_cap)

def _choose_mode(question: str, explicit: Optional[str]) -> str:
    if explicit in {"ml","ttest","literature"}:
        return explicit
    q = (question or "").lower()
    # If clearly an ML benchmarking question
    ml_hits = ["auc", "accuracy", "f1", "precision", "recall", "roc", "classifier", "model", "random forest", "logistic", "svm", "xgboost"]
    if any(w in q for w in ml_hits):
        return "ml"
    # If biomedical / causal / “risk of”, prefer literature review
    lit_hits = ["risk", "cause", "effect", "increase", "decrease", "treatment", "symptom", "prevalence",
                "meta-analysis", "systematic review", "cohort", "case-control", "trial", "cancer", "prostate", "breast"]
    if any(w in q for w in lit_hits):
        return "literature"
    # Otherwise: literature by default (safer than toy t-test)
    return "literature"

# ---------- API ----------
@app.post("/api/run")
def run(req: RunReq):
    run_id = f"run_{int(time.time()*1000)}"
    json_name = f"{run_id}.json"
    pdf_name  = f"{run_id}.pdf"
    json_path = ARTIFACT_DIR / json_name
    pdf_path  = ARTIFACT_DIR / pdf_name

    question = (req.question or "")[:500]
    mode = _choose_mode(question, req.experiment)

    # retrieval helpers
    sources = _get_sources(question, top_k_web=4, top_k_local=4, final_cap=6)

    # (optional) LLM summaries of sources
    summaries = []
    summarizer_id = None
    if os.getenv("ENABLE_LLM", "false").lower() == "true" and sources:
        try:
            sdata, smeta = summarize_sources_with_llm(sources, max_items=min(3, len(sources)))
            summaries = sdata.get("summaries", [])
            summarizer_id = f"{smeta.get('provider')}:{smeta.get('model')}"
        except Exception:
            pass

    # ---------- Branch: ML experiment ----------
    if mode == "ml":
        ml = run_ml_experiment()
        ds_meta = {"name": ml["dataset"]["name"], "target_name": ml["dataset"]["target_name"]}
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
                pass

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
                pass

        env_models = {"generator": "template-hypotheses-v1", "eval": "sklearn-1.5.1"}
        if summarizer_id:  env_models["summarizer"]  = summarizer_id
        if concluder_id:   env_models["concluder"]   = concluder_id
        if recommender_id: env_models["recommender"] = recommender_id

        report = {
            "schema": "amrra.report.v1",
            "run_id": run_id,
            "request": {"question": question, "mode": mode},
            "retrieval": {"sources": sources, "summaries": summaries},
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

    # ---------- Branch: Literature review (default) ----------
    else:
        # In literature mode we don’t fabricate stats; we ground in sources.
        ds_meta = {"name": "literature", "target_name": "evidence"}
        hyps = generate_hypotheses(question, ds_meta, sources)

        conclusion = {}
        concluder_id = None
        if os.getenv("ENABLE_LLM","false").lower() == "true":
            try:
                plan_for_llm = {"design": "literature_review"}
                results_for_llm = {"evidence_items": len(sources)}
                cdata, cmeta = build_conclusion_with_llm(question, plan_for_llm, results_for_llm, sources, hyps)
                conclusion = {
                    "text": cdata.get("conclusion"),
                    "strength": cdata.get("strength_of_evidence"),
                    "limitations": cdata.get("limitations"),
                    "next_steps": cdata.get("next_steps")
                }
                concluder_id = f"{cmeta.get('provider')}:{cmeta.get('model')}"
            except Exception:
                pass

        study_rec = {}
        recommender_id = None
        if os.getenv("ENABLE_LLM","false").lower() == "true":
            try:
                sdata, smeta = build_study_design_with_llm(
                    question=question,
                    plan={"design":"literature_review"},
                    results={"evidence_items": len(sources)},
                    sources=sources,
                    constraints={"budget": None, "compute": None, "deadline_days": 7}
                )
                study_rec = sdata
                recommender_id = f"{smeta.get('provider')}:{smeta.get('model')}"
            except Exception:
                pass

        env_models = {"generator": "template-hypotheses-v1"}
        if summarizer_id:  env_models["summarizer"]  = summarizer_id
        if concluder_id:   env_models["concluder"]   = concluder_id
        if recommender_id: env_models["recommender"] = recommender_id

        report = {
            "schema": "amrra.report.v1",
            "run_id": run_id,
            "request": {"question": question, "mode": mode},
            "retrieval": {"sources": sources, "summaries": summaries},
            "hypotheses": hyps,
            "plan": {"design": "literature_review"},
            "results": {"evidence_items": len(sources)},
            "conclusion": conclusion,
            "plan_recommendations": study_rec,
            "environment": {
                "datasets": [{"name": "literature", "hash": "sha256:dynamic-hydrated"}],
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

    # ---------- Canonical hash BEFORE writes ----------
    can = canonical_string(report)
    h = sha256_hex(can)
    report["report_hash"] = f"sha256:{h}"

    # ---------- Write JSON ----------
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Re-read & verify
    with open(json_path, "r", encoding="utf-8") as f:
        _disk = json.load(f)
    _disk_can = canonical_string(_disk)
    _disk_hash = "sha256:" + sha256_hex(_disk_can)
    if _disk_hash != report["report_hash"]:
        raise RuntimeError(f"Post-write rehash mismatch: disk={_disk_hash} computed={report['report_hash']}")

    # ---------- Render PDF ----------
    render_pdf(report, str(pdf_path))

    # ---------- (Optional) HCS anchor ----------
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
    return {"query": q, "results": RETRIEVER.search(q, top_k=k, web=WEB, web_k=k, hydrate=True)}

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
            pass
        out.append({
            "sequence_number": int(m.get("sequence_number", 0)),
            "consensus_timestamp": m.get("consensus_timestamp"),
            "payer_account_id": m.get("payer_account_id"),
            "message_decoded": decoded,
        })

    return {"topic_id": TOPIC_ID, "messages": out}
