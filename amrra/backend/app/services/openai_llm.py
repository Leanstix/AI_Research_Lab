# app/services/openai_llm.py
import os, hashlib
from typing import Dict, Any
from openai import OpenAI

_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env or export it in your shell."
        )
    return OpenAI(api_key=api_key)

async def generate_reasoning(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
    client = _get_client()
    messages = [
        {"role": "system", "content": "You are AMRRA, a transparent research assistant."},
        {"role": "user", "content": prompt},
    ]
    chat = client.chat.completions.create(model=_MODEL, messages=messages)
    content = chat.choices[0].message.content

    plan = {"prompt": prompt, "outline": content, "context": context or {}}
    return {
        "plan": plan,
        "plan_hash": _sha256_hex(str(plan)),
    }
