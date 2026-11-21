"""IR, multi-hop evidence completeness, and temporal retrieval evaluation metrics."""

from __future__ import annotations

import math
from datetime import date
from typing import Dict, Iterable, List, Sequence

import numpy as np

from chronoevidence.schemas import DocumentRecord, QueryRecord, RetrievedItem


def is_doc_temporally_valid(doc: DocumentRecord, query: QueryRecord) -> bool:
    """
    Deterministic temporal validity check against ground-truth clause effective intervals.
    - If query specifies `target_date`: doc must satisfy effective_from <= target_date <= effective_to (if set).
    - Else if query specifies `required_periods` (cross-period query): doc's effective_from year or
      publication_date year must belong to required_periods.
    - Otherwise: valid.
    """
    if query.target_date:
        target = date.fromisoformat(query.target_date[:10])
        eff_start = date.fromisoformat(doc.effective_from[:10])
        if target < eff_start:
            return False
        if doc.effective_to:
            eff_end = date.fromisoformat(doc.effective_to[:10])
            if target > eff_end:
                return False
        return True
    if query.required_periods:
        doc_years = {doc.effective_from[:4], doc.publication_date[:4]}
        return any(p in doc_years for p in query.required_periods)
    return True


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    gold_set = set(gold_ids)
    hits = sum(1 for d in retrieved_ids[:k] if d in gold_set)
    return hits / len(gold_set)


def hit_rate_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    gold_set = set(gold_ids)
    return 1.0 if any(d in gold_set for d in retrieved_ids[:k]) else 0.0


def mrr_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    gold_set = set(gold_ids)
    for rank, d in enumerate(retrieved_ids[:k], start=1):
        if d in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    gold_set = set(gold_ids)
    dcg = 0.0
    for rank, d in enumerate(retrieved_ids[:k], start=1):
        if d in gold_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evidence_completeness_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """1.0 iff EVERY required gold evidence document appears in top-k (strict multi-hop completeness)."""
    if not gold_ids:
        return float("nan")
    top_set = set(retrieved_ids[:k])
    return 1.0 if all(g in top_set for g in gold_ids) else 0.0


def temporal_precision_at_k(
    retrieved_ids: Sequence[str],
    query: QueryRecord,
    doc_map: Dict[str, DocumentRecord],
    k: int,
) -> float:
    """Fraction of top-k retrieved documents that are temporally valid for the query scope."""
    top_docs = retrieved_ids[:k]
    if not top_docs:
        return 0.0
    valid = sum(1 for d in top_docs if d in doc_map and is_doc_temporally_valid(doc_map[d], query))
    return valid / len(top_docs)


def temporal_coverage_at_k(
    retrieved_ids: Sequence[str],
    query: QueryRecord,
    doc_map: Dict[str, DocumentRecord],
    k: int,
) -> float:
    """
    Fraction of required temporal periods/versions covered by relevant/family-matched retrieved docs in top-k.
    Evaluated over queries with non-empty `required_periods` and `gold_doc_ids`.
    """
    if not query.required_periods or not query.gold_doc_ids:
        return float("nan")
    gold_families = {doc_map[g].family_id for g in query.gold_doc_ids if g in doc_map}
    covered = set()
    for d in retrieved_ids[:k]:
        if d not in doc_map:
            continue
        doc = doc_map[d]
        if doc.family_id not in gold_families:
            continue
        for p in query.required_periods:
            if doc.effective_from.startswith(p) or doc.publication_date.startswith(p):
                covered.add(p)
    return len(covered) / len(set(query.required_periods))


def decoy_intrusion_rate_at_k(retrieved_ids: Sequence[str], decoy_ids: Sequence[str], k: int) -> float:
    """1.0 if any known outdated/premature decoy appears in top-k (lower is better)."""
    if not decoy_ids:
        return float("nan")
    decoy_set = set(decoy_ids)
    return 1.0 if any(d in decoy_set for d in retrieved_ids[:k]) else 0.0


def evaluate_query(
    query: QueryRecord,
    retrieved: Sequence[RetrievedItem],
    doc_map: Dict[str, DocumentRecord],
) -> Dict[str, float]:
    ids = [r.doc_id for r in retrieved]
    return {
        "recall@1": recall_at_k(ids, query.gold_doc_ids, 1),
        "recall@3": recall_at_k(ids, query.gold_doc_ids, 3),
        "recall@5": recall_at_k(ids, query.gold_doc_ids, 5),
        "recall@10": recall_at_k(ids, query.gold_doc_ids, 10),
        "mrr@10": mrr_at_k(ids, query.gold_doc_ids, 10),
        "ndcg@5": ndcg_at_k(ids, query.gold_doc_ids, 5),
        "ndcg@10": ndcg_at_k(ids, query.gold_doc_ids, 10),
        "hit@1": hit_rate_at_k(ids, query.gold_doc_ids, 1),
        "hit@5": hit_rate_at_k(ids, query.gold_doc_ids, 5),
        "completeness@5": evidence_completeness_at_k(ids, query.gold_doc_ids, 5),
        "temporal_precision@5": temporal_precision_at_k(ids, query, doc_map, 5),
        "temporal_coverage@5": temporal_coverage_at_k(ids, query, doc_map, 5),
        "decoy_intrusion@3": decoy_intrusion_rate_at_k(ids, query.outdated_decoy_ids, 3),
    }


def aggregate_metrics(per_query_rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(per_query_rows)
    if not rows:
        return {}
    keys = list(rows[0].keys())
    summary: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in rows if not math.isnan(r[k])]
        summary[k] = float(np.mean(vals)) if vals else float("nan")
    return summary


def cross_check_with_ir_measures(
    queries: Sequence[QueryRecord],
    runs: Dict[str, Sequence[RetrievedItem]],
) -> Dict[str, float] | None:
    """Optional cross-check using `ir_measures` when installed in the environment."""
    try:
        import ir_measures
        from ir_measures import nDCG, R, RR
    except ImportError:
        return None

    qrels = []
    run_items = []
    for q in queries:
        if not q.gold_doc_ids:
            continue
        for g in q.gold_doc_ids:
            qrels.append(ir_measures.Qrel(q.query_id, g, 1))
        for item in runs.get(q.query_id, []):
            run_items.append(ir_measures.ScoredDoc(q.query_id, item.doc_id, item.score))

    res = ir_measures.calc_aggregate([R @ 5, R @ 10, RR @ 10, nDCG @ 5, nDCG @ 10], qrels, run_items)
    return {str(m): float(v) for m, v in res.items()}
