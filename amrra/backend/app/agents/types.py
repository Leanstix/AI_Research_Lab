from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class RetrievedDoc(BaseModel):
    id: str
    score: float = 0.0
    text: str
    meta: Dict[str, Any] = {}

class Extraction(BaseModel):
    key_facts: Dict[str, Any] = {}
    citations: List[Dict[str, Any]] = []

class Candidate(BaseModel):
    model: str
    content: str
    meta: Dict[str, Any] = {}

class Judgement(BaseModel):
    winner_index: int
    rationale: str
    scores: List[float]
