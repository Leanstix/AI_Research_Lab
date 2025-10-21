from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import Base, engine, get_db
from app.models import Job
from app.schemas import CreateJobRequest, CreateJobResponse, JobResponse
from app.tasks import run_research_job

# Ensure tables exist (or use Alembic in prod)
Base.metadata.create_all(bind=engine)

router = APIRouter()

@router.post("/", response_model=CreateJobResponse)
def create_job(payload: CreateJobRequest, db: Session = Depends(get_db)):
    job = Job(prompt=payload.prompt, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)

    # Enqueue async worker task
    run_research_job.delay(job_id=job.id, prompt=payload.prompt, context=payload.context or {})
    return CreateJobResponse(id=job.id, status=job.status)

@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobResponse(
        id=job.id,
        status=job.status,
        result=job.result,
        reason_trace_hash=job.reason_trace_hash,
        hcs_tx_id=job.hcs_tx_id,
    )
