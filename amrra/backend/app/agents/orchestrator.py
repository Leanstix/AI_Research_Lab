import json, hashlib, asyncio
from typing import Dict, Any
from app.agents.retriever import run_retriever
from app.agents.extractor import run_extractor
from app.agents.experimenter import run_experimentation
from app.agents.judge import run_judging

def _sha(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

async def run_agentic_pipeline(user_prompt: str, k: int = 6) -> Dict[str, Any]:
    # 1) retrieve
    docs = await run_retriever(user_prompt, k=k)
    # 2) extract
    extraction = await run_extractor(docs, user_prompt)
    # 3) experiment (parallel)
    candidates = await run_experimentation(user_prompt, extraction)
    # 4) judge
    judgement = await run_judging(user_prompt, candidates)

    final = candidates[judgement.winner_index].content if candidates else ""
    bundle = {
        "prompt": user_prompt,
        "retriever": [d.model_dump() for d in docs],
        "extraction": extraction.model_dump(),
        "candidates": [c.model_dump() for c in candidates],
        "judgement": judgement.model_dump(),
        "final": final,
    }
    return {
        "final": final,
        "bundle": bundle,
        "bundle_hash": _sha(json.dumps(bundle, ensure_ascii=False)),
    }
