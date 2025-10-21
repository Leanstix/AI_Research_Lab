from typing import List
from app.agents.types import Candidate, Judgement
from app.services.openai_llm import _get_client
import os

_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

async def run_judging(user_prompt: str, candidates: List[Candidate]) -> Judgement:
    client = _get_client()
    # build comparison prompt
    numbered = "\n\n".join([f"Candidate {i} ({c.model}):\n{c.content}" for i,c in enumerate(candidates)])
    sys = "You are the AMRRA Judging agent. Score each candidate for relevance, coherence, and grounding (0-10). Pick the single best index."
    usr = f"User task: {user_prompt}\n\n{numbered}\n\nReturn strictly JSON: {{\"scores\":[...],\"winner_index\":int,\"rationale\":str}}"
    resp = client.chat.completions.create(model=_MODEL, messages=[{"role":"system","content":sys},{"role":"user","content":usr}], temperature=0.0)
    import json
    txt = resp.choices[0].message.content or "{}"
    try:
        j = json.loads(txt)
        return Judgement(winner_index=int(j.get("winner_index", 0)), rationale=j.get("rationale",""), scores=[float(x) for x in j.get("scores", [])])
    except Exception:
        # fallback: choose first non-empty candidate
        idx = 0
        for i,c in enumerate(candidates):
            if (c.content or "").strip():
                idx = i; break
        return Judgement(winner_index=idx, rationale="fallback", scores=[0.0]*len(candidates))
