# app/core/webfetch.py
import trafilatura, re, html
from typing import Optional, Dict

def fetch_and_extract(url: str) -> Optional[Dict]:
    try:
        raw = trafilatura.fetch_url(url, no_ssl=True)
        if not raw:
            return None
        meta = trafilatura.extract(
            raw,
            include_comments=False,
            include_tables=False,
            output="json",
            favor_precision=True,
            with_metadata=True,
        )
        if not meta:
            return None
        data = trafilatura.extract(
            raw,
            include_comments=False,
            include_tables=False,
            output="txt",
            favor_precision=True,
            with_metadata=False,
        )
        j = trafilatura.utils.load_json(meta)
        title = (j.get("title") or "").strip()
        author = (j.get("author") or "").strip()
        date = (j.get("date") or "").strip()
        text = (data or "").strip()
        if not text:
            return None
        title = html.unescape(title)
        return {
            "title": title[:300] if title else None,
            "author": author or None,
            "date": date or None,
            "text": text,
            "url": url
        }
    except Exception:
        return None

def best_quote(text: str, query: str, max_chars: int = 280) -> str:
    sents = re.split(r'(?<=[.!?])\s+', text)
    if not sents:
        return text[:max_chars]
    q = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
    stop = {"the","a","an","and","of","to","for","in","on","at","with","is","are","be","as","by","that","this","it"}
    q = {w for w in q if w not in stop and len(w) > 2}
    scores = []
    for i, s in enumerate(sents):
        toks = set(re.findall(r"[a-z0-9]+", s.lower()))
        overlap = len(q & toks)
        scores.append((overlap, i))
    scores.sort(reverse=True)
    best_i = scores[0][1] if scores else 0
    window = " ".join(sents[best_i:best_i+2]).strip()
    if len(window) > max_chars:
        window = window[:max_chars].rsplit(" ", 1)[0] + "…"
    return window