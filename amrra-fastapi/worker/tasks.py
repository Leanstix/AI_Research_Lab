# worker/tasks.py
import os, json, pathlib, time
from typing import Optional
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

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
RUNS_DIR = BASE_DIR / "runs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

def _choose_mode(question: str, explicit: Optional[str]) -> str:
    if explicit in {"ml","ttest","literature"}:
        return explicit
    q = (question or "").lower()
    ml_hits = ["auc", "accuracy", "f1", "precision", "recall", "roc", "classifier", "model"]
    if any(w in q for w in ml_hits): return "ml"
    lit_hits = ["risk","effect","increase","decrease","treatment","meta-analysis","systematic","cohort","trial","cancer","prostate","breast"]
    if any(w in q for w in lit_hits): return "literature"
    return "literature"

def _dedupe_sources(items, max_k=6):
    seen = set(); out = []
    for s in items:
        key = (s.get("url") or (s.get("source") or {}).get("name") or s.get("title"))
        if not key: continue
        k = key.strip().lower()
        if k in seen: continue
        seen.add(k); out.append(s)
        if len(out) >= max_k: break
    return out

def _get_sources(q: str, corpus_dir: str):
    q_eff = (q or "").strip() or "baseline model comparison performance"
    web = WebSearch.from_env()
    try:
        web_sources = __import__("app.core.websearch", fromlist=["x"])
        web_hits = __import__("asyncio").get_event_loop().run_until_complete(web.search_and_summarize(q_eff, 4))
    except Exception:
        web_hits = []
    retriever = build_retriever(corpus_dir)
    try:
        local_hits = retriever.search(q_eff, top_k=4, web=web, web_k=4, hydrate=True)
    except Exception:
        local_hits = []
    return _dedupe_sources((web_hits or []) + (local_hits or []), max_k=6)

def run_pipeline(run_id: str, question: str, experiment: Optional[str] = None):
    """Main worker task. Writes manifest + artifacts and returns a summary dict."""
    env = os.environ
    corpus_dir = env.get("CORPUS_DIR", str(BASE_DIR / "corpus"))
    os.makedirs(corpus_dir, exist_ok=True)

    json_name = f"{run_id}.json"
    pdf_name  = f"{run_id}.pdf"
    json_path = ARTIFACT_DIR / json_name
    pdf_path  = ARTIFACT_DIR / pdf_name

    # 1) retrieval
    sources = _get_sources(question, corpus_dir)

    # 2) mode
    mode = _choose_mode(question, experiment)

    # 3) LLM summaries (optional)
    summaries = []
    summarizer_id = None
    if env.get("ENABLE_LLM","false").lower() == "true" and sources:
        try:
            sdata, smeta = summarize_sources_with_llm(sources, max_items=min(3, len(sources)))
            summaries = sdata.get("summaries", [])
            summarizer_id = f"{smeta.get('provider')}:{smeta.get('model')}"
        except Exception:
            pass

    # 4) branch
    if mode == "ml":
        ml = run_ml_experiment()
        ds_meta = {"name": ml["dataset"]["name"], "target_name": ml["dataset"]["target_name"]}
        hyps = generate_hypotheses(question, ds_meta, sources)

        conclusion = {}; concluder_id = None
        if env.get("ENABLE_LLM","false").lower() == "true":
            try:
                plan_for_llm = {"design": ml["design"]["task"], "primary_metric": ml["design"]["primary_metric"], "test_size": ml["design"]["test_size"], "seed": ml["design"]["seed"]}
                results_for_llm = {"dataset": ml["dataset"], "metrics": ml["metrics"], "comparison": ml["comparison"]}
                cdata, cmeta = build_conclusion_with_llm(question, plan_for_llm, results_for_llm, sources, hyps)
                conclusion = {"text": cdata.get("conclusion"), "strength": cdata.get("strength_of_evidence"), "limitations": cdata.get("limitations"), "next_steps": cdata.get("next_steps")}
                concluder_id = f"{cmeta.get('provider')}:{cmeta.get('model')}"
            except Exception:
                pass

        env_models = {"generator": "template-hypotheses-v1", "eval": "sklearn-1.5.1"}
        if summarizer_id: env_models["summarizer"]=summarizer_id
        if concluder_id: env_models["concluder"]=concluder_id

        report = {
            "schema": "amrra.report.v1",
            "run_id": run_id,
            "request": {"question": question, "mode": mode},
            "retrieval": {"sources": sources, "summaries": summaries},
            "hypotheses": hyps,
            "plan": {"design": ml["design"]["task"], "primary_metric": ml["design"]["primary_metric"], "test_size": ml["design"]["test_size"], "seed": ml["design"]["seed"]},
            "results": {"dataset": ml["dataset"], "metrics": ml["metrics"], "comparison": ml["comparison"]},
            "conclusion": conclusion,
            "environment": {"datasets": [{"name": ml["dataset"]["name"], "hash": "sha256:sklearn-canonical"}], "code": {"image": "local-demo"}, "models": env_models, "hardware": "cpu"},
            "artifacts": {"json": {"type": "local", "name": json_name}, "pdf": {"type": "local", "name": pdf_name}},
            "timestamp_unix": int(time.time()),
            "report_hash": None
        }
    else:
        ds_meta = {"name": "literature", "target_name": "evidence"}
        hyps = generate_hypotheses(question, ds_meta, sources)
        conclusion = {}; concluder_id = None
        if env.get("ENABLE_LLM","false").lower() == "true":
            try:
                plan_for_llm = {"design": "literature_review"}
                results_for_llm = {"evidence_items": len(sources)}
                cdata, cmeta = build_conclusion_with_llm(question, plan_for_llm, results_for_llm, sources, hyps)
                conclusion = {"text": cdata.get("conclusion"), "strength": cdata.get("strength_of_evidence"), "limitations": cdata.get("limitations"), "next_steps": cdata.get("next_steps")}
                concluder_id = f"{cmeta.get('provider')}:{cmeta.get('model')}"
            except Exception:
                pass

        env_models = {"generator": "template-hypotheses-v1"}
        if summarizer_id: env_models["summarizer"]=summarizer_id
        if concluder_id: env_models["concluder"]=concluder_id

        report = {
            "schema": "amrra.report.v1",
            "run_id": run_id,
            "request": {"question": question, "mode": mode},
            "retrieval": {"sources": sources, "summaries": summaries},
            "hypotheses": hyps,
            "plan": {"design": "literature_review"},
            "results": {"evidence_items": len(sources)},
            "conclusion": conclusion,
            "environment": {"datasets": [{"name": "literature", "hash": "sha256:dynamic-hydrated"}], "code": {"image": "local-demo"}, "models": env_models, "hardware": "cpu"},
            "artifacts": {"json": {"type": "local", "name": json_name}, "pdf": {"type": "local", "name": pdf_name}},
            "timestamp_unix": int(time.time()),
            "report_hash": None
        }

    # hash
    can = canonical_string(report)
    h = sha256_hex(can)
    report["report_hash"] = f"sha256:{h}"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    render_pdf(report, str(pdf_path))

    # update manifest
    manifest = {
        "run_id": run_id,
        "request": {"question": question, "experiment": experiment},
        "status": "completed",
        "artifacts": {
            "json": f"/api/artifacts/{json_name}",
            "pdf": f"/api/artifacts/{pdf_name}"
        },
        "report_hash": report["report_hash"],
        "ts": int(time.time())
    }
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return manifest