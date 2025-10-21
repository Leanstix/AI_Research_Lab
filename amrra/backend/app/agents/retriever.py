from typing import List
from app.agents.types import RetrievedDoc
from app.services.rust_client import similar

async def run_retriever(query: str, k: int = 6) -> List[RetrievedDoc]:
    hits = await similar(query, k=k)  # optional rust service
    if not hits:
        # graceful fallback: no external retriever available
        return [RetrievedDoc(id="fallback-0", score=0.0, text=query, meta={"note":"no retriever available"})]
    docs = []
    for i, h in enumerate(hits):
        text = h.get("text") or h.get("content") or ""
        docs.append(RetrievedDoc(id=str(h.get("id", i)), score=float(h.get("score", 0.0)), text=text, meta=h))
    return docs
