"""Hybrid retrieval via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from typing import Dict, List, Sequence

from chronoevidence.retrieval.base import BaseRetriever
from chronoevidence.schemas import DocumentRecord, RetrievedItem


class HybridRRFRetriever(BaseRetriever):
    """Reciprocal Rank Fusion (Cormack et al., 2009) over multiple base retrievers."""

    def __init__(
        self,
        retrievers: Sequence[BaseRetriever],
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
        candidate_depth: int = 50,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRRFRetriever requires at least one base retriever.")
        self.name = "hybrid_rrf"
        self.retrievers = list(retrievers)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.retrievers)
        if len(self.weights) != len(self.retrievers):
            raise ValueError("Length of weights must match length of retrievers.")
        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth

    def index(self, documents: Sequence[DocumentRecord]) -> None:
        for ret in self.retrievers:
            ret.index(documents)

    def retrieve(self, query: str, top_k: int = 10, target_date: str | None = None) -> List[RetrievedItem]:
        depth = max(self.candidate_depth, top_k)
        fused_scores: Dict[str, float] = {}
        comp_ranks: Dict[str, Dict[str, int]] = {}
        comp_scores: Dict[str, Dict[str, float]] = {}

        for ret, weight in zip(self.retrievers, self.weights):
            hits = ret.retrieve(query, top_k=depth, target_date=target_date)
            for item in hits:
                fused_scores[item.doc_id] = fused_scores.get(item.doc_id, 0.0) + weight / (self.rrf_k + item.rank)
                comp_ranks.setdefault(item.doc_id, {})[ret.name] = item.rank
                comp_scores.setdefault(item.doc_id, {})[ret.name] = item.score

        ranked_ids = sorted(fused_scores.keys(), key=lambda d: (-fused_scores[d], d))[:top_k]
        return [
            RetrievedItem(
                doc_id=doc_id,
                rank=rank,
                score=float(fused_scores[doc_id]),
                method=self.name,
                component_ranks=comp_ranks[doc_id],
                component_scores=comp_scores[doc_id],
            )
            for rank, doc_id in enumerate(ranked_ids, start=1)
        ]

    def get_metadata(self) -> Dict[str, object]:
        return {
            "retriever": self.name,
            "rrf_k": self.rrf_k,
            "candidate_depth": self.candidate_depth,
            "weights": self.weights,
            "components": [r.get_metadata() for r in self.retrievers],
        }
