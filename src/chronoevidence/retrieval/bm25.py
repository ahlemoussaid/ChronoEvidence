"""BM25 sparse retrieval baselines (exact Okapi BM25 + optional Pyserini adapter)."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from chronoevidence.retrieval.base import BaseRetriever
from chronoevidence.schemas import DocumentRecord, RetrievedItem

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
        "was", "were", "which", "with", "this", "these", "those", "what", "does",
    }
)


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        return [t for t in tokens if t not in _STOPWORDS]
    return tokens


class BM25OkapiRetriever(BaseRetriever):
    """Deterministic Robertson-Sparck Jones Okapi BM25 implementation (no external binary required)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, remove_stopwords: bool = True) -> None:
        self.name = "bm25_okapi"
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self.doc_ids: List[str] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Counter[str]] = []
        self.doc_lengths: np.ndarray = np.array([], dtype=np.float64)
        self.avgdl: float = 0.0

    def index(self, documents: Sequence[DocumentRecord]) -> None:
        self.doc_ids = [d.doc_id for d in documents]
        self.doc_term_freqs = []
        self.doc_freqs = {}
        lengths: List[int] = []

        for doc in documents:
            full_text = f"{doc.title} {doc.text}"
            toks = tokenize(full_text, remove_stopwords=self.remove_stopwords)
            tf = Counter(toks)
            self.doc_term_freqs.append(tf)
            lengths.append(len(toks))
            for term in tf:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        n_docs = len(documents)
        self.doc_lengths = np.asarray(lengths, dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_lengths)) if n_docs > 0 else 0.0

        # Non-negative Lucene-compatible BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
        self.idf = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freqs.items()
        }

    def score_all(self, query: str) -> np.ndarray:
        q_terms = tokenize(query, remove_stopwords=self.remove_stopwords)
        scores = np.zeros(len(self.doc_ids), dtype=np.float64)
        if not q_terms or len(self.doc_ids) == 0 or self.avgdl == 0.0:
            return scores

        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf_map in enumerate(self.doc_term_freqs):
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lengths[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[idx] += idf * ((tf * (self.k1 + 1.0)) / denom)
        return scores

    def retrieve(self, query: str, top_k: int = 10, target_date: str | None = None) -> List[RetrievedItem]:
        scores = self.score_all(query)
        # Deterministic descending score, ascending doc_id tie-break
        order = sorted(range(len(self.doc_ids)), key=lambda i: (-float(scores[i]), self.doc_ids[i]))[:top_k]
        return [
            RetrievedItem(
                doc_id=self.doc_ids[idx],
                rank=rank,
                score=float(scores[idx]),
                method=self.name,
                component_ranks={self.name: rank},
                component_scores={self.name: float(scores[idx])},
            )
            for rank, idx in enumerate(order, start=1)
        ]

    def get_metadata(self) -> Dict[str, object]:
        return {
            "retriever": self.name,
            "k1": self.k1,
            "b": self.b,
            "remove_stopwords": self.remove_stopwords,
            "corpus_size": len(self.doc_ids),
            "avgdl": round(self.avgdl, 3),
        }


class PyseriniBM25Retriever(BaseRetriever):
    """Optional Pyserini Lucene BM25 adapter for large-scale local/GPU benchmark runs."""

    def __init__(self, index_dir: str, k1: float = 0.9, b: float = 0.4) -> None:
        self.name = "bm25_pyserini"
        self.index_dir = index_dir
        self.k1 = k1
        self.b = b
        self._searcher = None

    def index(self, documents: Sequence[DocumentRecord]) -> None:
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as exc:
            raise RuntimeError(
                "Pyserini is not installed in this environment. Install requirements-full.txt "
                "or use BM25OkapiRetriever."
            ) from exc
        self._searcher = LuceneSearcher(self.index_dir)
        self._searcher.set_bm25(k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 10, target_date: str | None = None) -> List[RetrievedItem]:
        if self._searcher is None:
            raise RuntimeError("Call index() before retrieve().")
        hits = self._searcher.search(query, k=top_k)
        return [
            RetrievedItem(
                doc_id=hit.docid,
                rank=rank,
                score=float(hit.score),
                method=self.name,
                component_ranks={self.name: rank},
                component_scores={self.name: float(hit.score)},
            )
            for rank, hit in enumerate(hits, start=1)
        ]
