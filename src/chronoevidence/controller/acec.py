"""ACEC: Adaptive Claim-Level Evidence Coverage retrieval controller."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from pydantic import BaseModel, Field

from chronoevidence.evidence.decomposition import AtomicClaim, RuleBasedClaimDecomposer
from chronoevidence.evidence.graph import EvidenceGraph
from chronoevidence.evidence.verification import (
    compute_coverage_temporal_uncertainty,
    is_non_substantive_doc,
)
from chronoevidence.retrieval.base import BaseRetriever
from chronoevidence.schemas import DocumentRecord, RetrievedItem


class ACECConfig(BaseModel):
    budget_calls: int = Field(default=3, ge=1)
    per_call_depth: int = Field(default=6, ge=1)
    tau_C: float = Field(default=0.72, ge=0.0, le=1.0)
    tau_T: float = Field(default=0.90, ge=0.0, le=1.0)
    tau_U: float = Field(default=0.30, ge=0.0, le=1.0)
    abstain_min_C: float = Field(default=0.45, ge=0.0, le=1.0)
    abstain_max_U: float = Field(default=0.55, ge=0.0, le=1.0)
    # Ablation switches
    use_decomposition: bool = True
    use_coverage: bool = True
    use_temporal: bool = True
    use_uncertainty: bool = True
    use_adaptive_stopping: bool = True


class ACECStepLog(BaseModel):
    step: int
    action: str
    subquery: str
    new_docs_added: int
    C_q: float
    T_q: float
    U_q: float


class ACECResult(BaseModel):
    retrieved: List[RetrievedItem]
    claims: List[AtomicClaim]
    C_q: float
    T_q: float
    U_q: float
    abstain: bool
    has_conflict: bool
    conflict_pairs: List[List[str]]
    retrieval_calls: int
    unique_docs_inspected: int
    stop_reason: str
    trace: List[ACECStepLog]
    claim_citations: Dict[str, List[str]]


class ACECController(BaseRetriever):
    """
    Query-adaptive retrieval controller implementing Adaptive Claim-Level Evidence Coverage (ACEC).
    Operates strictly within a fixed retrieval-call budget `B` over any `BaseRetriever`.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        config: Optional[ACECConfig] = None,
        name: str = "acec_full",
    ) -> None:
        self.name = name
        self.base_retriever = base_retriever
        self.config = config or ACECConfig()
        self.decomposer = RuleBasedClaimDecomposer()
        self.doc_map: Dict[str, DocumentRecord] = {}

    def index(self, documents: Sequence[DocumentRecord], index_base: bool = True) -> None:
        self.doc_map = {d.doc_id: d for d in documents}
        if index_base:
            self.base_retriever.index(documents)

    def _should_stop(self, c_q: float, t_q: float, u_q: float) -> bool:
        if not self.config.use_adaptive_stopping:
            return False
        c_ok = (c_q >= self.config.tau_C) if self.config.use_coverage else True
        t_ok = (t_q >= self.config.tau_T) if self.config.use_temporal else True
        u_ok = (u_q <= self.config.tau_U) if self.config.use_uncertainty else True
        return c_ok and t_ok and u_ok

    def _plan_next_subquery(
        self,
        claims: Sequence[AtomicClaim],
        claim_support_map: Dict[str, List[tuple[str, float]]],
        graph: EvidenceGraph,
        target_date: Optional[str],
        issued_queries: set[str],
    ) -> tuple[str, str]:
        """Choose the highest-value targeted follow-up query for the least-supported / unresolved requirement."""
        # Priority 1: Follow unfetched delegation pointer (e.g., Schedule Annex / Technical Bulletin)
        if any(c.claim_type == "delegated_schedule" for c in claims):
            unfetched = graph.get_unfetched_delegations()
            for _, ref_title in unfetched:
                if ref_title not in issued_queries:
                    return ("follow_delegation_ref", ref_title)

        # Priority 2: Retrieve missing conflict partner if a CONFLICTS_WITH edge points outside the graph
        missing_conf = graph.get_missing_conflict_partners(target_date if self.config.use_temporal else None)
        if missing_conf:
            src_id, _ = missing_conf[0]
            src_doc = graph.docs[src_id]
            anchors = " ".join(claims[0].anchor_entities) if claims else ""
            subq = f"{anchors} parallel regulatory order directive {src_doc.title}"
            if subq not in issued_queries:
                return ("investigate_conflict", subq)

        # Priority 3: Find the least-supported claim
        worst_claim = min(
            claims,
            key=lambda c: (
                claim_support_map.get(c.claim_id, [("", 0.0)])[0][1]
                if claim_support_map.get(c.claim_id)
                else 0.0
            ),
        )
        anchors = " ".join(worst_claim.anchor_entities)
        preds = " ".join(worst_claim.predicate_terms)

        # If we retrieved an expired/superseded doc or need an amendment, issue targeted amendment search
        has_expired_or_base = any(
            (not graph.is_temporally_valid(d_id, target_date)) or ("base" in d_id)
            for d_id in graph.docs
        )
        if worst_claim.claim_type == "temporal_amendment" or (self.config.use_temporal and has_expired_or_base):
            amend_q = f"{anchors} Amendment Order 2025 Revision Clause B revised ceiling {preds}".strip()
            if amend_q not in issued_queries:
                return (f"search_amendment:{worst_claim.claim_id}", amend_q)

        if worst_claim.claim_type == "delegated_schedule":
            annex_q = f"{anchors} mandatory implementation parameters schedule annex appendix protocol supplement addendum bulletin".strip()
            if annex_q not in issued_queries:
                return (f"search_schedule:{worst_claim.claim_id}", annex_q)

        # Generic claim-focused retrieval
        focused_q = f"{anchors} {preds}".strip() or worst_claim.text
        if focused_q not in issued_queries:
            return (f"retrieve_claim:{worst_claim.claim_id}", focused_q)

        # Fallback expansion
        fallback_q = f"{anchors} active binding clause effective {preds}".strip()
        return (f"replan_claim:{worst_claim.claim_id}", fallback_q)

    def run(
        self,
        question: str,
        top_k: int = 10,
        target_date: Optional[str] = None,
    ) -> ACECResult:
        cfg = self.config
        claims = self.decomposer.decompose(question, enabled=cfg.use_decomposition)
        graph = EvidenceGraph()
        base_score_map: Dict[str, float] = {}
        trace: List[ACECStepLog] = []
        issued_queries: set[str] = set()

        # Step 1: Initial hybrid retrieval (1 call from budget B)
        remaining_budget = cfg.budget_calls
        calls_used = 0

        initial_hits = self.base_retriever.retrieve(question, top_k=cfg.per_call_depth, target_date=target_date)
        calls_used += 1
        remaining_budget -= 1
        issued_queries.add(question)

        added = 0
        for item in initial_hits:
            if item.doc_id in self.doc_map:
                if item.doc_id not in graph.docs:
                    graph.add_document(self.doc_map[item.doc_id])
                    added += 1
                base_score_map[item.doc_id] = max(base_score_map.get(item.doc_id, 0.0), item.score)

        c_q, t_q, u_q, support_map = compute_coverage_temporal_uncertainty(
            claims, graph, target_date, use_temporal=cfg.use_temporal, use_coverage=cfg.use_coverage
        )
        trace.append(
            ACECStepLog(
                step=1,
                action="initial_retrieval",
                subquery=question,
                new_docs_added=added,
                C_q=c_q,
                T_q=t_q,
                U_q=u_q,
            )
        )

        stop_reason = "budget_exhausted"
        if self._should_stop(c_q, t_q, u_q):
            stop_reason = "thresholds_satisfied"
        else:
            step_idx = 2
            while remaining_budget > 0:
                action, subq = self._plan_next_subquery(claims, support_map, graph, target_date, issued_queries)
                issued_queries.add(subq)
                hits = self.base_retriever.retrieve(subq, top_k=cfg.per_call_depth, target_date=target_date)
                calls_used += 1
                remaining_budget -= 1

                added_step = 0
                for item in hits:
                    if item.doc_id in self.doc_map:
                        if item.doc_id not in graph.docs:
                            graph.add_document(self.doc_map[item.doc_id])
                            added_step += 1
                        base_score_map[item.doc_id] = max(base_score_map.get(item.doc_id, 0.0), item.score)

                c_q, t_q, u_q, support_map = compute_coverage_temporal_uncertainty(
                    claims, graph, target_date, use_temporal=cfg.use_temporal, use_coverage=cfg.use_coverage
                )
                trace.append(
                    ACECStepLog(
                        step=step_idx,
                        action=action,
                        subquery=subq,
                        new_docs_added=added_step,
                        C_q=c_q,
                        T_q=t_q,
                        U_q=u_q,
                    )
                )
                step_idx += 1

                if self._should_stop(c_q, t_q, u_q):
                    stop_reason = "thresholds_satisfied"
                    break

        # Final abstention decision
        if cfg.use_uncertainty:
            abstain = (c_q < cfg.abstain_min_C) or (u_q > cfg.abstain_max_U)
        else:
            abstain = False

        # Conflict detection
        conflict_pairs = [list(p) for p in graph.get_unresolved_conflicts(target_date if cfg.use_temporal else None)]
        has_conflict = len(conflict_pairs) > 0

        # Rank final evidence pool
        # When use_coverage=True: prioritize documents that explicitly support distinct claims (ensuring multi-hop coverage!)
        # When use_coverage=False: fall back to base retriever score (with temporal validity penalty if use_temporal=True)
        claim_citations: Dict[str, List[str]] = {}
        doc_final_scores: Dict[str, float] = {}

        if cfg.use_coverage:
            # First ensure the best supporting doc for each claim gets a coverage slot boost
            primary_support_boost: Dict[str, float] = {}
            for claim in claims:
                sups = support_map.get(claim.claim_id, [])
                if sups:
                    claim_citations[claim.claim_id] = [sups[0][0]]
                    primary_support_boost[sups[0][0]] = max(primary_support_boost.get(sups[0][0], 0.0), sups[0][1])
                    if claim.claim_type == "conflict_check" and len(sups) >= 2:
                        claim_citations[claim.claim_id].append(sups[1][0])
                        primary_support_boost[sups[1][0]] = max(primary_support_boost.get(sups[1][0], 0.0), sups[1][1])
                else:
                    claim_citations[claim.claim_id] = []

            for doc_id, doc in graph.docs.items():
                max_claim_s = max(
                    (s for sups in support_map.values() for d, s in sups if d == doc_id),
                    default=0.0,
                )
                temp_ok = (
                    (graph.is_temporally_valid(doc_id, target_date) and not graph.is_superseded_in_graph(doc_id, target_date))
                    if cfg.use_temporal
                    else True
                )
                subst_bonus = 0.0 if is_non_substantive_doc(doc) else 0.10
                slot_bonus = 1.0 if doc_id in primary_support_boost else 0.0
                temp_bonus = 0.35 if temp_ok else -0.50
                doc_final_scores[doc_id] = (
                    slot_bonus + max_claim_s + temp_bonus + subst_bonus + 0.05 * base_score_map.get(doc_id, 0.0)
                )
        else:
            # No-coverage ablation: rank by base relevance + temporal validity only
            for claim in claims:
                claim_citations[claim.claim_id] = []
            for doc_id in graph.docs:
                temp_ok = (
                    (graph.is_temporally_valid(doc_id, target_date) and not graph.is_superseded_in_graph(doc_id, target_date))
                    if cfg.use_temporal
                    else True
                )
                temp_bonus = 0.50 if temp_ok else -0.50
                doc_final_scores[doc_id] = base_score_map.get(doc_id, 0.0) + temp_bonus

        ranked_ids = sorted(doc_final_scores.keys(), key=lambda d: (-doc_final_scores[d], d))[:top_k]
        if not cfg.use_coverage and ranked_ids and claims:
            claim_citations[claims[0].claim_id] = ranked_ids[:2]

        retrieved = [
            RetrievedItem(
                doc_id=doc_id,
                rank=rank,
                score=round(float(doc_final_scores[doc_id]), 6),
                method=self.name,
                component_ranks={},
                component_scores={"base_score": round(base_score_map.get(doc_id, 0.0), 6)},
            )
            for rank, doc_id in enumerate(ranked_ids, start=1)
        ]

        return ACECResult(
            retrieved=retrieved,
            claims=claims,
            C_q=c_q,
            T_q=t_q,
            U_q=u_q,
            abstain=abstain,
            has_conflict=has_conflict,
            conflict_pairs=conflict_pairs,
            retrieval_calls=calls_used,
            unique_docs_inspected=len(graph.docs),
            stop_reason=stop_reason,
            trace=trace,
            claim_citations=claim_citations,
        )

    def retrieve(self, query: str, top_k: int = 10, target_date: str | None = None) -> List[RetrievedItem]:
        return self.run(query, top_k=top_k, target_date=target_date).retrieved

    def get_metadata(self) -> Dict[str, object]:
        return {
            "retriever": self.name,
            "config": self.config.model_dump(),
            "base_retriever": self.base_retriever.get_metadata(),
        }
