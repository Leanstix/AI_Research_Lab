# app/core/retrieval.py
import os, re, glob, json, unicodedata, asyncio, time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from app.core.hashcanon import sha256_hex
from app.core.webfetch import fetch_and_extract
from app.core.websearch import WebSearch

_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    path: Optional[str] = None
    doc_hash: Optional[str] = None

class TfidfRetriever:
    def __init__(self, docs: List[Doc], max_features: int = 20000, max_chars_index: int = 200_000, corpus_dir: Optional[str] = None):
        """
        Build a TF-IDF index from (optionally) truncated document texts to keep memory in check.
        The original full text is preserved on each Doc for quoting and hashing.
        """
        self.docs = docs
        self.corpus_dir = corpus_dir
        self.max_chars_index = int(max_chars_index)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            lowercase=True,
            min_df=1,
            max_df=0.95,
        )
        if self.docs:
            self._index_texts: List[str] = [_clean_text(d.text)[: self.max_chars_index] for d in docs]
            self.X = self.vectorizer.fit_transform(self._index_texts)
            for d in self.docs:
                d.doc_hash = "sha256:" + sha256_hex(d.text)
        else:
            # Fit a tiny placeholder vocabulary so .transform() works before hydration
            self._index_texts = []
            self.vectorizer.fit(["seed"])
            vocab_size = len(self.vectorizer.get_feature_names_out())
            self.X = sp.csr_matrix((0, vocab_size), dtype=float)

    def _best_snippet(self, text: str, q: str, width_chars: int = 220) -> Tuple[str, Tuple[int,int]]:
        # sentence-level scoring (reuse same vectorizer vocab)
        try:
            sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
            if not sents:
                return (text[:width_chars], (0, min(len(text), width_chars)))
            # compute per-sentence score vs query
            Q = self.vectorizer.transform([q])
            S = self.vectorizer.transform(sents)
            scores = (S @ Q.T).toarray().ravel()
            best_i = int(np.argmax(scores))
            # try 1–2 sentence window
            window = " ".join(sents[best_i: best_i+2])
            if len(window) > width_chars:
                window = window[:width_chars].rsplit(" ", 1)[0] + "…"
            # char range (approximate, guard against -1)
            start = text.find(sents[best_i])
            if start < 0:
                return (window, (0, min(len(text), len(window))))
            end = start + len(sents[best_i])
            return (window, (start, min(end, len(text))))
        except Exception:
            return (text[:width_chars], (0, min(len(text), width_chars)))

    def search(self, query: str, top_k: int = 3, web: Optional[WebSearch] = None, web_k: int = 5, hydrate: bool = True) -> List[Dict[str, Any]]:
        """Search local memory; if empty or OOV AND web provided, hydrate from the web, persist, reindex, and retry."""
        query = (query or "").strip()
        if not query:
            query = "baseline model comparison performance"

        # If we have zero docs, try hydrating first
        if not self.docs and hydrate and web:
            self._web_hydrate_and_reindex(query, web, k=max(web_k, top_k))
            if not self.docs:
                return []

        qv = self.vectorizer.transform([query])
        # cosine similarity (X and qv are L2-normalized by TF-IDF)
        sims = (self.X @ qv.T).toarray().ravel() if self.X.shape[0] > 0 and qv.nnz > 0 else np.zeros(len(self.docs), dtype=float)
        # Fallback when all similarities are zero (OOV query terms)
        if not np.any(sims):
            sims = self._fallback_scores(query)
        # If still all zero and we are allowed to hydrate, go fetch from web, persist, re-index, and retry
        if hydrate and web and not np.any(sims):
            added = self._web_hydrate_and_reindex(query, web, k=max(web_k, top_k))
            if added > 0:
                # retry with updated index
                qv = self.vectorizer.transform([query])
                sims = (self.X @ qv.T).toarray().ravel() if self.X.shape[0] > 0 and qv.nnz > 0 else np.zeros(len(self.docs), dtype=float)
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
                "doc_hash": d.doc_hash
            })
        return out

    def _fallback_scores(self, query: str) -> np.ndarray:
        """Lightweight token-overlap backstop when TF-IDF yields all zeros (e.g., OOV query terms)."""
        q_tokens = _tokens(query)
        if not q_tokens or not self.docs:
            return np.zeros(len(self.docs), dtype=float)
        scores: List[float] = []
        for d in self.docs:
            head = (_clean_text(d.text)[: self.max_chars_index]).lower()
            hay_tokens = set(re.findall(r"[a-z0-9]+", (d.title + " " + head)))
            scores.append(float(len(q_tokens & hay_tokens)))
        arr = np.array(scores, dtype=float)
        m = arr.max() if len(arr) else 0.0
        return arr / m if m > 0 else arr

    # --- Web hydration & persistence ---
    def _web_hydrate_and_reindex(self, query: str, web: WebSearch, k: int = 5) -> int:
        """Search the web, extract pages, persist new docs (if corpus_dir), and rebuild the TF-IDF index."""
        try:
            hits = asyncio.run(web.search(query, k=k))
        except RuntimeError:
            # If inside a running loop (rare for sync routes), try again (same call is fine here)
            hits = asyncio.run(web.search(query, k=k))

        new_docs: List[Doc] = []
        existing_hashes = {d.doc_hash for d in self.docs if d.doc_hash}
        for h in hits:
            page = fetch_and_extract(h.get("url",""))
            if not page or not page.get("text"):
                continue
            text = _clean_text(page["text"])
            if not text:
                continue
            doc_hash = "sha256:" + sha256_hex(text)
            if doc_hash in existing_hashes:
                continue
            title = (page.get("title") or h.get("title") or "Untitled").strip()
            # persist if corpus_dir available
            path = None
            if self.corpus_dir:
                os.makedirs(self.corpus_dir, exist_ok=True)
                stem = _sanitize_filename(title) or ("doc_" + str(int(time.time()*1000)))
                txt_path = os.path.join(self.corpus_dir, f"{stem}.txt")
                meta_path = os.path.join(self.corpus_dir, f"{stem}.json")
                _safe_write(txt_path, text)
                _safe_write(meta_path, json.dumps({
                    "title": title, "url": page.get("url"), "author": page.get("author"),
                    "date": page.get("date"), "doc_hash": doc_hash
                }, ensure_ascii=False, indent=2))
                path = txt_path
            # in-memory doc
            doc_id = f"web_{len(self.docs) + len(new_docs)}"
            new_docs.append(Doc(doc_id=doc_id, title=title, text=text, path=path, doc_hash=doc_hash))
            existing_hashes.add(doc_hash)

        if not new_docs:
            return 0

        # extend docs and rebuild index
        self.docs.extend(new_docs)
        self._index_texts = [_clean_text(d.text)[: self.max_chars_index] for d in self.docs]
        self.X = self.vectorizer.fit_transform(self._index_texts)
        return len(new_docs)

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

def _fallback_docs() -> List[Doc]:
    d1 = Doc(
        doc_id="mem_1",
        title="Why Random Forests Often Beat Logistic Regression",
        text=("Tree ensembles capture non-linear feature interactions and can yield higher ROC AUC than linear baselines "
              "on structured datasets. Regularization and proper calibration remain important for deployment.")
    )
    d2 = Doc(
        doc_id="mem_2",
        title="ROC AUC and Imbalanced Classes",
        text=("ROC AUC is threshold-independent and robust for imbalanced labels. Accuracy can be misleading when classes "
              "are skewed. Always report multiple metrics such as AUC, F1, and confusion matrix.")
    )
    d3 = Doc(
        doc_id="mem_3",
        title="Bootstrap Confidence Intervals",
        text=("Non-parametric bootstrap provides uncertainty estimates for metrics such as AUC. Reporting 95% intervals "
              "alongside point estimates improves reproducibility and scientific rigor.")
    )
    return [d1,d2,d3]

def build_retriever(corpus_dir: Optional[str]) -> TfidfRetriever:
    docs = _load_files_from_dir(corpus_dir)
    # If nothing on disk yet, start empty so web hydration can truly populate memory
    return TfidfRetriever(docs, corpus_dir=corpus_dir)

def _clean_text(s: str) -> str:
    """Unicode-normalize, strip control chars, and collapse whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    # remove control chars except common whitespace
    s = "".join(ch for ch in s if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    # collapse whitespace
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s+\n", "\n\n", s)
    return s.strip()

def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def _sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"[^\w\-. ]+", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name).strip("._")
    return name[:100] or "doc"

def _safe_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)
