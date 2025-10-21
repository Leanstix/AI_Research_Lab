from typing import List
from app.agents.types import RetrievedDoc, Extraction
from app.services.openai_llm import _get_client
import os

_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

async def run_extractor(docs: List[RetrievedDoc], user_prompt: str) -> Extraction:
    if not docs:
        return Extraction(key_facts={"note": "no docs"}, citations=[])
    client = _get_client()
    joined = "\n\n".join([f"[{i}] {d.text}" for i, d in enumerate(docs)])
    sys = "Extract the most relevant structured facts from the provided passages in compact JSON. Include citation indices."
    usr = f"User goal: {user_prompt}\n\nPassages:\n{joined}\n\nReturn JSON with keys: key_facts (object), citations (array of {{index, reason}})."
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or "{}"
    # best-effort parse; keep raw if not valid JSON
    import json
    try:
        j = json.loads(content)
        return Extraction(key_facts=j.get("key_facts", {}), citations=j.get("citations", []))
    except Exception:
        return Extraction(key_facts={"raw": content}, citations=[])
