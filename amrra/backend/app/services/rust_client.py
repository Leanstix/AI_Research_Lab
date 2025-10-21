import os, httpx
RUST_SIM_BASE = os.getenv("RUST_SIM_BASE")

async def preprocess(text: str) -> str:
    if not RUST_SIM_BASE:
        return text
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{RUST_SIM_BASE}/preprocess", json={"text": text})
        r.raise_for_status()
        return r.json().get("text", text)

async def similar(query: str, k: int = 8):
    if not RUST_SIM_BASE:
        return []
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f"{RUST_SIM_BASE}/similar", params={"q": query, "k": k})
        r.raise_for_status()
        return r.json().get("hits", [])
