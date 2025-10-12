# app/core/retrieval.py
import os, re, glob, json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from app.core.hashcanon import sha256_hex

_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    path: Optional[str] = None
    doc_hash: Optional[str] = None

class TfidfRetriever:
    def __init__(self, docs: List[Doc], max_features: int = 20000):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1,2),
            max_features=max_features,
            lowercase=True
        )
        corpus = [d.text for d in docs]
        self.X = self.vectorizer.fit_transform(corpus)
        # precompute doc hashes
        for d in self.docs:
            d.doc_hash = "sha256:" + sha256_hex(d.text)

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
            # char range (approximate)
            start = text.find(sents[best_i])
            end = start + len(sents[best_i])
            return (window, (max(start,0), max(end,0)))
        except Exception:
            return (text[:width_chars], (0, min(len(text), width_chars)))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not query:
            query = "baseline model comparison performance"
        qv = self.vectorizer.transform([query])
        # cosine similarity (X and qv are L2-normalized by TF-IDF)
        sims = (self.X @ qv.T).toarray().ravel()
        idx = np.argsort(-sims)[:top_k]
        out = []
        for i in idx:
            d = self.docs[int(i)]
            snippet, (s,e) = self._best_snippet(d.text, query)
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
                text = f.read()
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
    if not docs:
        docs = _fallback_docs()
    return TfidfRetriever(docs)
