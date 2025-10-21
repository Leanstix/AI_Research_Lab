from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.sql import func
from app.db import Base

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    status = Column(String(32), default="queued", index=True)
    prompt = Column(Text, nullable=False)
    result = Column(JSON, nullable=True)
    reason_trace_hash = Column(String(128), nullable=True)  # sha256 hex
    hcs_tx_id = Column(String(128), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Artifact(Base):
    __tablename__ = "artifacts"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, index=True)
    kind = Column(String(32))  # "plan" | "analysis" | "summary" | etc.
    content = Column(Text)
    sha256 = Column(String(128))
