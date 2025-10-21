from typing import List, Dict, Any
from app.agents.types import Extraction, Candidate
from app.services.openai_llm import _get_client
from app.services.ollama_llm import chat as ollama_chat
import os, asyncio

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_OLLAMA_MODELS = [m.strip() for m in os.getenv("OLLAMA_MODELS", "llama3.1:8b,qwen2.5:7b").split(",") if m.strip()]

def _base_messages(user_prompt: str, facts: Dict[str, Any]):
    sys = "You are AMRRA Experimentation agent. Produce a concise, high-quality candidate answer grounded in provided facts when possible."
    usr = f"Task: {user_prompt}\nGrounded facts (may be partial): {facts}"
    return [{"role":"system","content":sys},{"role":"user","content":usr}]

async def _openai_candidate(user_prompt: str, facts: Dict[str, Any]) -> Candidate:
    client = _get_client()
    msgs = _base_messages(user_prompt, facts)
    resp = client.chat.completions.create(model=_OPENAI_MODEL, messages=msgs, temperature=0.5)
    return Candidate(model=f"openai:{_OPENAI_MODEL}", content=resp.choices[0].message.content or "", meta={})

async def _ollama_candidate(model: str, user_prompt: str, facts: Dict[str, Any]) -> Candidate:
    msgs = _base_messages(user_prompt, facts)
    try:
        txt = await ollama_chat(msgs, model=model)
        return Candidate(model=f"ollama:{model}", content=txt or "", meta={})
    except Exception as e:
        return Candidate(model=f"ollama:{model}", content=f"[ollama unavailable: {e}]", meta={"error": str(e)})

async def run_experimentation(user_prompt: str, extraction: Extraction) -> List[Candidate]:
    tasks = [_openai_candidate(user_prompt, extraction.key_facts)]
    # try multiple local models in parallel if configured
    for m in _OLLAMA_MODELS:
        tasks.append(_ollama_candidate(m, user_prompt, extraction.key_facts))
    return await asyncio.gather(*tasks)
