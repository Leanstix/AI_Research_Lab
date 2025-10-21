# app/core/retrieval.py
import os, re, glob, json, unicodedata
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from app.core.hashcanon import sha256_hex
from app.core.webfetch import fetch_and_extract

_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    path: Optional[str] = None
    doc_hash: Optional[str] = None
    url: Optional[str] = None

class TfidfRetriever:
    def __init__(self, docs: List[Doc], corpus_dir: Optional[str] = None, max_features: int = 20000, max_chars_index: int = 200_000):
        self.docs = docs
        self.corpus_dir = corpus_dir
        self.max_chars_index = int(max_chars_index)
        self._index_texts: List[str] = [ _clean_text(d.text)[: self.max_chars_index] for d in docs ]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            lowercase=True,
            min_df=1,
            max_df=0.95,
        )
        self.X = self.vectorizer.fit_transform(self._index_texts)
        for d in self.docs:
            d.doc_hash = "sha256:" + sha256_hex(d.text)

    def search(self, query: str, top_k: int = 3, *, web=None, web_k: int = 3, hydrate: bool = False) -> List[Dict[str, Any]]:
        if not query:
            query = "baseline model comparison performance"
        query = (query or "").strip() or "baseline model comparison performance"
        out = self._search_once(query, top_k)
        if hydrate and not out and web is not None:
            hits = []
            try:
                hits = asyncio_run(web.search(query, k=max(3, web_k)))
            except Exception:
                hits = []
            added = 0
            for h in hits:
                page = fetch_and_extract(h.get("url",""))
                if not page or not page.get("text"):
                    continue
                title = page.get("title") or h.get("title") or "Untitled"
                text = _clean_text(page["text"])
                self._persist_and_add(title, text, page.get("url"))
                added += 1
                if added >= web_k:
                    break
            if added:
                self._refit_index()
                out = self._search_once(query, top_k)
        return out

    def _search_once(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not len(self.docs):
            return []
        qv = self.vectorizer.transform([query])
        sims = (self.X @ qv.T).toarray().ravel() if qv.nnz > 0 else np.zeros(len(self.docs), dtype=float)
        if not np.any(sims):
            sims = self._fallback_scores(query)
        idx = np.argsort(-sims)[:max(0, min(top_k, len(self.docs)))]
        out = []
        for i in idx:
            d = self.docs[int(i)]
            snippet, (s, e) = self._best_snippet(d.text, query)
            out.append({
                "doc_id": d.doc_id,
                "title": d.title,
                "score": float(sims[int(i)]),
                "quote": snippet,
                "char_range": [int(s), int(e)],
                "source": (
                    {"type":"file","name": os.path.basename(d.path), "path": d.path}
                    if d.path else {"type":"inmem","name": d.title}
                ),
                "doc_hash": d.doc_hash,
                "url": d.url
            })
        return out

    def _best_snippet(self, text: str, q: str, width_chars: int = 220) -> Tuple[str, Tuple[int,int]]:
        try:
            sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
            if not sents:
                return (text[:width_chars], (0, min(len(text), width_chars)))
            Q = self.vectorizer.transform([q])
            S = self.vectorizer.transform(sents)
            scores = (S @ Q.T).toarray().ravel()
            best_i = int(np.argmax(scores))
            window = " ".join(sents[best_i: best_i+2])
            if len(window) > width_chars:
                window = window[:width_chars].rsplit(" ", 1)[0] + "…"
            start = text.find(sents[best_i])
            if start < 0:
                return (window, (0, min(len(text), len(window))))
            end = start + len(sents[best_i])
            return (window, (start, min(end, len(text))))
        except Exception:
            return (text[:width_chars], (0, min(len(text), width_chars)))

    def _fallback_scores(self, query: str) -> np.ndarray:
        q_tokens = _tokens(query)
        if not q_tokens or not self._index_texts:
            return np.zeros(len(self.docs), dtype=float)
        scores: List[float] = []
        for d, head in zip(self.docs, self._index_texts):
            hay = (d.title + " \n " + head).lower()
            hay_tokens = set(re.findall(r"[a-z0-9]+", hay))
            overlap = len(q_tokens & hay_tokens)
            scores.append(float(overlap))
        import numpy as np
        arr = np.array(scores, dtype=float)
        m = arr.max() if len(arr) else 0.0
        return arr / m if m > 0 else arr

    def _persist_and_add(self, title: str, text: str, url: Optional[str] = None):
        if not text:
            return
        if self.corpus_dir:
            os.makedirs(self.corpus_dir, exist_ok=True)
            h = sha256_hex(text)
            safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", title)[:80] or "doc"
            base = f"{safe}_{h[:10]}"
            txt_path = os.path.join(self.corpus_dir, base + ".txt")
            meta_path = os.path.join(self.corpus_dir, base + ".json")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"title": title, "url": url, "sha256": h}, f, ensure_ascii=False, indent=2)
            d = Doc(doc_id=f"doc_{len(self.docs)}", title=title, text=text, path=txt_path, url=url)
        else:
            d = Doc(doc_id=f"doc_{len(self.docs)}", title=title, text=text, path=None, url=url)
        d.doc_hash = "sha256:" + sha256_hex(d.text)
        self.docs.append(d)
        self._index_texts.append(_clean_text(text)[: self.max_chars_index])

    def _refit_index(self):
        self.X = self.vectorizer.fit_transform(self._index_texts)

def _load_files_from_dir(corpus_dir: str, exts=(".txt",".md")) -> List[Doc]:
    docs: List[Doc] = []
    if not corpus_dir or not os.path.isdir(corpus_dir):
        return docs
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(corpus_dir, f"**/*{ext}"), recursive=True)
    for n, p in enumerate(sorted(paths)):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            text = _clean_text(raw)
            if not text:
                continue
            title = os.path.basename(p)
            docs.append(Doc(doc_id=f"doc_{n}", title=title, text=text, path=p))
        except Exception:
            continue
    return docs

def build_retriever(corpus_dir: Optional[str]) -> TfidfRetriever:
    docs = _load_files_from_dir(corpus_dir)
    return TfidfRetriever(docs, corpus_dir=corpus_dir)

def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = "".join(ch for ch in s if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s+\n", "\n\n", s)
    return s.strip()

def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def asyncio_run(coro):
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        import threading
        result = {}
        def runner():
            result["v"] = asyncio.new_event_loop().run_until_complete(coro)
        t = threading.Thread(target=runner)
        t.start(); t.join()
        return result.get("v")
    else:
        return loop.run_until_complete(coro)