import os, httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "").rstrip("/")

async def chat(messages, model: str) -> str:
    if not OLLAMA_URL:
        raise RuntimeError("OLLAMA_URL not set")
    payload = {"model": model, "messages": messages, "stream": False}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        j = r.json()
        # newer ollama chat returns {'message':{'content': '...'}}
        m = j.get("message", {}).get("content")
        if m:
            return m
        # older format fallback
        return j.get("response") or ""
