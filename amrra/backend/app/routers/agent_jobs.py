from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from sqlalchemy.orm import Session
from app.db import Base, engine, get_db
from app.models import Job, Artifact
from app.tasks import run_agentic_job

Base.metadata.create_all(bind=engine)
router = APIRouter()

class AgentJobCreate(BaseModel):
    prompt: str = Field(..., min_length=3)
    k: int = 6
    context: Optional[Dict[str, Any]] = None

class AgentJobStatus(BaseModel):
    id: int
    status: str

@router.post("/", response_model=AgentJobStatus)
def create_agent_job(payload: AgentJobCreate, db: Session = Depends(get_db)):
    job = Job(prompt=payload.prompt, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    run_agentic_job.delay(job_id=job.id, prompt=payload.prompt, k=payload.k, context=payload.context or {})
    return AgentJobStatus(id=job.id, status=job.status)
