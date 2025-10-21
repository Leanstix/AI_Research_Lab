import hashlib, json, asyncio
from celery import shared_task
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models import Job, Artifact
from app.services.openai_llm import generate_reasoning
from app.services.rust_client import preprocess
from app.services.hcs_client import log_to_hcs
from app.agents.orchestrator import run_agentic_pipeline

def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

@shared_task
def run_research_job(job_id: int, prompt: str, context: dict):
    db: Session = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if not job:
            return

        job.status = "running"
        db.commit()

        # === Pipeline ===
        # 1) Optional fast preprocess via Rust microservice
        processed_prompt = asyncio.run(preprocess(prompt))

        # 2) Generate reasoning/plan via your AI stack (kept same)
        reasoning = asyncio.run(generate_reasoning(processed_prompt, context))
        plan_json = json.dumps(reasoning["plan"], ensure_ascii=False)

        # 3) Hash artifacts
        plan_hash = reasoning["plan_hash"]
        trace_hash = _sha256_hex(plan_json)  # include more artifacts if needed

        art = Artifact(job_id=job_id, kind="plan", content=plan_json, sha256=plan_hash)
        db.add(art)
        db.commit()

        # 4) Log to Hedera HCS via Node service
        hcs_res = asyncio.run(log_to_hcs(
            message=plan_json,
            memo="AMRRA reasoning trace",
            reference_id=str(job_id),
            hash_hex=trace_hash,
        ))

        # 5) Persist result
        job.status = "succeeded"
        job.result = {"plan": reasoning["plan"]}
        job.reason_trace_hash = trace_hash
        job.hcs_tx_id = hcs_res.get("txId")
        db.commit()

    except Exception as e:
        if job := db.get(Job, job_id):
            job.status = "failed"
            job.result = {"error": str(e)}
            db.commit()
    finally:
        db.close()
        
@shared_task
def run_agentic_job(job_id: int, prompt: str, k: int = 6, context: dict = None):
    db: Session = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if not job:
            return
        job.status = "running"
        db.commit()

        result = asyncio.run(run_agentic_pipeline(prompt, k=k))
        bundle = result["bundle"]
        bundle_json = json.dumps(bundle, ensure_ascii=False)
        bundle_hash = result["bundle_hash"]

        # store artifacts by stage
        db.add(Artifact(job_id=job_id, kind="agent:retriever", content=json.dumps(bundle["retriever"]), sha256=_sha256_hex(json.dumps(bundle["retriever"]))))
        db.add(Artifact(job_id=job_id, kind="agent:extraction", content=json.dumps(bundle["extraction"]), sha256=_sha256_hex(json.dumps(bundle["extraction"]))))
        db.add(Artifact(job_id=job_id, kind="agent:candidates", content=json.dumps(bundle["candidates"]), sha256=_sha256_hex(json.dumps(bundle["candidates"]))))
        db.add(Artifact(job_id=job_id, kind="agent:judgement", content=json.dumps(bundle["judgement"]), sha256=_sha256_hex(json.dumps(bundle["judgement"]))))
        db.add(Artifact(job_id=job_id, kind="agent:final", content=bundle["final"], sha256=_sha256_hex(bundle["final"])))
        db.commit()

        # Log compact on-chain message (hash + job id)
        hcs_res = asyncio.run(log_to_hcs(
            message=json.dumps({"referenceId": job_id, "bundle_hash": bundle_hash}),
            memo="AMRRA agentic",
            reference_id=str(job_id),
            hash_hex=bundle_hash,
        ))
        job.hcs_tx_id = (hcs_res or {}).get("txId")

        job.status = "succeeded"
        job.result = {"final": bundle["final"], "summary": {"winner_model": bundle["judgement"]["winner_index"], "hash": bundle_hash}}
        job.reason_trace_hash = bundle_hash
        db.commit()

    except Exception as e:
        if job := db.get(Job, job_id):
            job.status = "failed"
            job.result = {"error": str(e)}
            db.commit()
    finally:
        db.close()
