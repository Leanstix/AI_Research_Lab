from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class CreateJobRequest(BaseModel):
    prompt: str = Field(..., min_length=3)
    context: Optional[Dict[str, Any]] = None

class JobResponse(BaseModel):
    id: int
    status: str
    result: Optional[Dict[str, Any]] = None
    reason_trace_hash: Optional[str] = None
    hcs_tx_id: Optional[str] = None

class CreateJobResponse(BaseModel):
    id: int
    status: str
