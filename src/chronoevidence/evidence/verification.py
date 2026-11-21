"""Explicit claim-level evidence support verification, coverage C(q), temporal consistency T(q), and uncertainty U(q)."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from chronoevidence.evidence.decomposition import AtomicClaim
from chronoevidence.evidence.graph import EvidenceGraph
from chronoevidence.retrieval.bm25 import tokenize
from chronoevidence.schemas import DocumentRecord

# Textual indicators that a document is non-binding commentary, press summary, draft, or filing instructions
_NON_SUBSTANTIVE_CUES = (
    "no regulatory amendments",
    "non-binding draft",
    "does not establish binding",
    "refer to substantive clauses",
    "electronic compliance portal",
    "soliciting stakeholder comments",
)

# Textual indicators of substantive binding content (numbers, limits, obligations, procedures)
_SUBSTANTIVE_CUE_RE = re.compile(
    r"(\b\d+(?:\.\d+)?\b|\b10\^\d+\b|\bmust\b|\bmandates\b|\bstrictly\b|\btightened\b|\brevised\b|\binstructed\b|\bexempt\b|\bpermitted\b|\bdeferred\b|\bzero\b)"
)


def is_non_substantive_doc(doc: DocumentRecord) -> bool:
    low = f"{doc.title} {doc.text}".lower()
    return any(cue in low for cue in _NON_SUBSTANTIVE_CUES)


def score_claim_support(
    claim: AtomicClaim,
    doc: DocumentRecord,
    graph: EvidenceGraph,
    target_date: Optional[str],
    use_temporal: bool = True,
) -> float:
    """
    Explicitly evaluate whether `doc` substantively supports `claim` in [0.0, 1.0].
    Does NOT rely on raw retrieval rank alone:
      1. Penalizes non-substantive drafts/commentaries/press releases/filing forms.
      2. Requires entity/anchor alignment when anchors are present.
      3. Evaluates distinctive predicate-term recall (excluding generic boilerplate).
      4. Handles provenance-linked amendments (MODIFIES/SUPERSEDES) and delegated schedules.
      5. Applies temporal validity when `use_temporal=True`.
    """
    # 1. Temporal validity gate (if enabled)
    if use_temporal:
        if not graph.is_temporally_valid(doc.doc_id, target_date):
            return 0.0
        if graph.is_superseded_in_graph(doc.doc_id, target_date):
            return 0.0

    # 2. Non-substantive document penalty
    if is_non_substantive_doc(doc):
        return 0.05

    full_text = f"{doc.title} {doc.text}"
    full_low = full_text.lower()
    doc_tokens = set(tokenize(full_text, remove_stopwords=True))

    # 3. Entity anchor check (prevents cross-domain hijack)
    if claim.anchor_entities:
        anchor_hit = any(a.lower() in full_low for a in claim.anchor_entities)
        if not anchor_hit:
            return 0.0

    # 4. Special handling for delegated_schedule claims
    if claim.claim_type == "delegated_schedule":
        # A document supports the delegated schedule claim if it is referenced by a retrieved amendment's delegation pointer
        for src_id, refs in graph.delegation_pointers.items():
            if src_id in graph.docs and (not use_temporal or graph.is_temporally_valid(src_id, target_date)):
                for ref in refs:
                    if ref.lower() in doc.title.lower() or doc.title.lower() in ref.lower():
                        return 0.95
        # Fallback: direct annex/schedule substance match
        if any(k in doc.title.lower() for k in ("annex", "schedule", "appendix", "protocol", "supplement", "addendum", "bulletin")):
            if "mandatory implementation parameters" in full_low or "compliance engineering" in full_low:
                return 0.85
        return 0.0

    # 5. Special handling for conflict_check claims
    if claim.claim_type == "conflict_check":
        conflicts = graph.get_unresolved_conflicts(target_date if use_temporal else None)
        involved = {d for pair in conflicts for d in pair}
        if doc.doc_id in involved:
            # Verify topic alignment too
            pred_terms = claim.predicate_terms
            overlap = sum(1 for t in pred_terms if t in doc_tokens) / max(1, len(pred_terms))
            return min(1.0, 0.5 + 0.5 * overlap)
        return 0.0

    # 6. Distinctive predicate-term coverage
    pred_terms = claim.predicate_terms
    if not pred_terms:
        pred_overlap = 0.5
    else:
        matched = sum(1 for t in pred_terms if t in doc_tokens)
        pred_overlap = matched / len(pred_terms)

    # 7. Provenance-aware amendment support:
    # If `doc` MODIFIES or SUPERSEDES a retrieved document that strongly matches the claim's predicate terms
    # AND `doc` explicitly revises that clause (e.g., terse amendment referring to "Clause B"),
    # inherit predicate alignment from the superseded/modified clause.
    inherited_overlap = 0.0
    if doc.modifies or doc.supersedes:
        for old_id in list(doc.modifies) + list(doc.supersedes):
            if old_id in graph.docs:
                old_doc = graph.docs[old_id]
                old_tokens = set(tokenize(f"{old_doc.title} {old_doc.text}", remove_stopwords=True))
                old_ov = sum(1 for t in pred_terms if t in old_tokens) / max(1, len(pred_terms))
                # Check that the claim asks for the revised/amended clause (not an unmodified clause A)
                # Specifically: if old_doc has a superseded guidance/FAQ focused on that clause, or claim is temporal_amendment
                if old_id in doc.supersedes and old_ov >= 0.5:
                    inherited_overlap = max(inherited_overlap, old_ov)
                elif claim.claim_type == "temporal_amendment" and old_ov >= 0.5:
                    inherited_overlap = max(inherited_overlap, old_ov * 0.95)

    effective_overlap = max(pred_overlap, inherited_overlap)

    # Require genuine predicate support (prevents unanswerable queries with only entity overlap from passing!)
    if effective_overlap < 0.35:
        return 0.0

    # Substance check (binding numeric/obligation content)
    has_substance = bool(_SUBSTANTIVE_CUE_RE.search(full_low))
    substance_factor = 1.0 if has_substance else 0.65

    # If `doc` is modified by an active amendment AND the claim is asking for the amended/revised clause,
    # downweight `doc` in favor of the amendment while keeping `doc` strong for unmodified factual claims!
    if use_temporal and claim.claim_type == "temporal_amendment":
        active_mods = graph.get_active_modifiers_of(doc.doc_id, target_date)
        if active_mods:
            return round(0.25 * effective_overlap * substance_factor, 4)

    return round(min(1.0, effective_overlap * substance_factor), 4)


def compute_coverage_temporal_uncertainty(
    claims: Sequence[AtomicClaim],
    graph: EvidenceGraph,
    target_date: Optional[str],
    use_temporal: bool = True,
    use_coverage: bool = True,
) -> Tuple[float, float, float, Dict[str, List[Tuple[str, float]]]]:
    """
    Compute:
      - C(q) = sum(w_i * s_i) / sum(w_i)
      - T(q) in [0, 1]: temporal consistency of the best supporting evidence set
      - U(q) in [0, 1]: unresolved uncertainty combining coverage deficit, temporal ambiguity, and unresolved gaps
      - claim_support_map: {claim_id: [(doc_id, score), ...]} sorted descending
    """
    claim_support_map: Dict[str, List[Tuple[str, float]]] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    selected_support_ids: List[str] = []

    for claim in claims:
        scored_docs: List[Tuple[str, float]] = []
        for doc_id, doc in graph.docs.items():
            s = score_claim_support(claim, doc, graph, target_date, use_temporal=use_temporal)
            if s > 0.0:
                scored_docs.append((doc_id, s))
        scored_docs.sort(key=lambda x: (-x[1], x[0]))
        claim_support_map[claim.claim_id] = scored_docs

        if use_coverage:
            best_s = scored_docs[0][1] if scored_docs else 0.0
            # For conflict_check claims, full support requires BOTH conflicting authorities to be present
            if claim.claim_type == "conflict_check":
                if len(scored_docs) >= 2:
                    best_s = min(1.0, 0.5 * (scored_docs[0][1] + scored_docs[1][1]))
                else:
                    best_s = 0.35 * best_s
        else:
            # Ablation: without claim-level coverage, treat any retrieved non-empty pool as constant nominal support
            best_s = 1.0 if graph.docs else 0.0

        weighted_sum += claim.weight * best_s
        total_weight += claim.weight
        if scored_docs:
            selected_support_ids.append(scored_docs[0][0])
            if claim.claim_type == "conflict_check" and len(scored_docs) >= 2:
                selected_support_ids.append(scored_docs[1][0])

    c_q = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    # Temporal consistency T(q): fraction of selected supporting docs (or top-3 graph docs if none) that are
    # temporally valid AND not superseded as of target_date
    eval_ids = list(dict.fromkeys(selected_support_ids)) if selected_support_ids else list(graph.docs.keys())[:3]
    if not eval_ids:
        t_q = 0.0
    elif not use_temporal:
        t_q = 1.0
    else:
        valid_cnt = sum(
            1
            for d_id in eval_ids
            if graph.is_temporally_valid(d_id, target_date)
            and not graph.is_superseded_in_graph(d_id, target_date)
        )
        t_q = valid_cnt / len(eval_ids)

    # Unresolved uncertainty U(q):
    # - coverage deficit (1 - C(q))
    # - temporal inconsistency (1 - T(q))
    # - unresolved structural pointers (unfetched delegations or missing conflict partners)
    # - version-staleness risk: if an expired document in the same family is retrieved alongside an old
    #   unmodified base doc, and no newer amendment modifying that base doc has been inspected yet.
    unfetched_del = len(graph.get_unfetched_delegations()) if any(c.claim_type == "delegated_schedule" for c in claims) else 0
    missing_conf = len(graph.get_missing_conflict_partners(target_date if use_temporal else None))
    structural_penalty = 0.25 if (unfetched_del > 0 or missing_conf > 0) else 0.0

    staleness_penalty = 0.0
    if use_temporal and target_date and selected_support_ids:
        expired_families = {
            d.family_id
            for d_id, d in graph.docs.items()
            if not graph.is_temporally_valid(d_id, target_date)
        }
        has_amendment_claim = any(c.claim_type == "temporal_amendment" for c in claims)
        for d_id in selected_support_ids:
            doc = graph.docs[d_id]
            # If the supporting doc belongs to a family that has an expired sibling AND no active modifier
            # has been retrieved for it yet, flag staleness uncertainty so the controller verifies amendments.
            if (
                has_amendment_claim
                and doc.family_id in expired_families
                and not doc.modifies
                and not doc.supersedes
                and not graph.get_active_modifiers_of(d_id, target_date)
            ):
                staleness_penalty = 0.35
                break

    u_q = min(1.0, 0.60 * (1.0 - c_q) + 0.25 * (1.0 - t_q) + structural_penalty + staleness_penalty)

    return round(c_q, 4), round(t_q, 4), round(u_q, 4), claim_support_map
