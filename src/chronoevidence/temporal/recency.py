"""Temporal-aware retrieval baselines (Publication-Date Recency Boost + Naive Pub-Date Filter)."""

from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Sequence

from chronoevidence.retrieval.base import BaseRetriever
from chronoevidence.schemas import DocumentRecord, RetrievedItem


def parse_iso_date(date_str: str) -> date:
    return date.fromisoformat(date_str[:10])


class TemporalRecencyRetriever(BaseRetriever):
    """
    Temporal-aware baseline combining base relevance with exponential publication-date recency decay.
    Reflects standard recency-prioritized retrieval (cf. Re3 freshness intuition) without clause-level
    validity intervals or multi-period preservation.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        alpha: float = 0.65,
        decay_lambda: float = 0.35,
        reference_date: str = "2026-11-20",
        candidate_depth: int = 40,
    ) -> None:
        self.name = "temporal_recency_hybrid"
        self.base_retriever = base_retriever
        self.alpha = alpha
        self.decay_lambda = decay_lambda
        self.reference_date = reference_date
        self.candidate_depth = candidate_depth
        self.doc_map: Dict[str, DocumentRecord] = {}

    def index(self, documents: Sequence[DocumentRecord], index_base: bool = True) -> None:
        self.doc_map = {d.doc_id: d for d in documents}
        if index_base:
            self.base_retriever.index(documents)

    def retrieve(self, query: str, top_k: int = 10, target_date: str | None = None) -> List[RetrievedItem]:
        ref_dt = parse_iso_date(target_date or self.reference_date)
        candidates = self.base_retriever.retrieve(query, top_k=max(self.candidate_depth, top_k), target_date=target_date)
        if not candidates:
            return []

        max_s = max(c.score for c in candidates)
        min_s = min(c.score for c in candidates)
        denom = (max_s - min_s) if max_s > min_s else 1.0

        rescored: List[tuple[str, float, RetrievedItem, float]] = []
        for item in candidates:
            doc = self.doc_map[item.doc_id]
            pub_dt = parse_iso_date(doc.publication_date)
            # Standard recency baseline penalizes documents published after target_date if target_date is given,
            # and decays older documents exponentially by age in years.
            delta_days = (ref_dt - pub_dt).days
            if delta_days < 0:
                recency_score = math.exp(-2.0 * abs(delta_days) / 365.25)
            else:
                age_years = delta_days / 365.25
                recency_score = math.exp(-self.decay_lambda * age_years)

            norm_rel = (item.score - min_s) / denom
            combined = self.alpha * norm_rel + (1.0 - self.alpha) * recency_score
            rescored.append((item.doc_id, combined, item, recency_score))

        rescored.sort(key=lambda x: (-x[1], x[0]))
        results: List[RetrievedItem] = []
        for rank, (doc_id, score, base_item, rec_s) in enumerate(rescored[:top_k], start=1):
            c_scores = dict(base_item.component_scores)
            c_scores["base_score"] = base_item.score
            c_scores["recency_score"] = round(rec_s, 6)
            results.append(
                RetrievedItem(
                    doc_id=doc_id,
                    rank=rank,
                    score=float(score),
                    method=self.name,
                    component_ranks={**base_item.component_ranks, self.base_retriever.name: base_item.rank},
                    component_scores=c_scores,
                )
            )
        return results

    def get_metadata(self) -> Dict[str, object]:
        return {
            "retriever": self.name,
            "alpha": self.alpha,
            "decay_lambda": self.decay_lambda,
            "reference_date": self.reference_date,
            "candidate_depth": self.candidate_depth,
            "base": self.base_retriever.get_metadata(),
        }
