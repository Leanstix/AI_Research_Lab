# app/core/websearch.py
from typing import List, Dict, Optional
import os, httpx
from urllib.parse import urlparse, parse_qs, unquote
from bs4 import BeautifulSoup
from app.core.webfetch import fetch_and_extract, best_quote
from app.core.hashcanon import sha256_hex

class WebSearch:
    def __init__(self, provider: str, api_key: Optional[str] = None, cse_id: Optional[str] = None):
        self.provider = (provider or "duckduckgo").lower()
        self.api_key = api_key
        self.cse_id = cse_id

    @staticmethod
    def from_env():
        prov = os.getenv("SEARCH_PROVIDER", "duckduckgo")
        return WebSearch(
            provider=prov,
            api_key=os.getenv("GOOGLE_API_KEY"),
            cse_id=os.getenv("GOOGLE_CSE_ID")
        )

    async def _search_duckduckgo(self, q: str, k: int) -> List[Dict]:
        # Use HTML endpoint (no API key). Respect site ToS; this is a simple, light fetch.
        url = "https://duckduckgo.com/html/"
        params = {"q": q}
        out = []
        headers = {"User-Agent": "Mozilla/5.0 AMRRA/1.0"}
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.select(".result__a")[:k]:
                href = self._ddg_resolve(a.get("href"))
                title = a.get_text(" ", strip=True)
                if href and title:
                    out.append({"title": title, "url": href})
        return out

    @staticmethod
    def _ddg_resolve(href: Optional[str]) -> Optional[str]:
        if not href:
            return href
        try:
            u = urlparse(href)
            if u.netloc.endswith("duckduckgo.com"):
                qs = parse_qs(u.query)
                uddg = qs.get("uddg", [None])[0]
                if uddg:
                    return unquote(uddg)
        except Exception:
            pass
        return href

    async def _search_google_cse(self, q: str, k: int) -> List[Dict]:
        if not self.api_key or not self.cse_id:
            return []
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": self.api_key, "cx": self.cse_id, "q": q, "num": min(k, 10)}
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        out = []
        for item in (data.get("items") or [])[:k]:
            out.append({"title": item.get("title"), "url": item.get("link")})
        return out

    async def search(self, q: str, k: int = 5) -> List[Dict]:
        if self.provider == "google":
            res = await self._search_google_cse(q, k)
            if res:
                return res
            # fallback to ddg if google empty/misconfigured
        return await self._search_duckduckgo(q, k)

    async def search_and_summarize(self, q: str, k: int = 3) -> List[Dict]:
        hits = await self.search(q, k)
        results = []
        for rank, h in enumerate(hits, start=1):
            page = fetch_and_extract(h["url"])
            if not page:
                continue
            quote = best_quote(page["text"], q)
            doc_hash = "sha256:" + sha256_hex(page["text"])
            results.append({
                "title": page["title"] or h["title"],
                "url": page["url"],
                "quote": quote,
                "rank": rank,
                "doc_hash": doc_hash,
                "author": page.get("author"),
                "date": page.get("date"),
                "source": {"type": "web", "name": page["url"]}
            })
        return results
