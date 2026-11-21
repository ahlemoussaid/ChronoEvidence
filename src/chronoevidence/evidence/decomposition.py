"""Query-to-atomic-claim decomposition for ACEC."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from chronoevidence.retrieval.bm25 import tokenize

_ACRONYM_RE = re.compile(r"\b[A-Z]{3,8}\b")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

# Generic regulatory/question boilerplate excluded when extracting distinctive predicate terms
_BOILERPLATE_TERMS = frozenset(
    {
        "under", "following", "revision", "amendment", "rule", "rules", "requirement",
        "requirements", "ceiling", "limit", "limits", "exact", "statutory", "binding",
        "currently", "legally", "enforceable", "applicable", "apply", "jointly",
        "updated", "governs", "governing", "regarding", "between", "authorities",
        "regulatory", "there", "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december", "2021",
        "2022", "2023", "2024", "2025", "2026", "2027", "which", "what", "when",
        "how", "does", "required", "parameters", "technical", "verification",
        "delegated", "schedule", "unresolved", "conflict",
    }
)


class AtomicClaim(BaseModel):
    claim_id: str
    text: str
    weight: float = Field(default=1.0, gt=0.0)
    claim_type: str = Field(
        default="factual",
        description="factual | delegated_schedule | conflict_check | temporal_amendment",
    )
    anchor_entities: List[str] = Field(default_factory=list)
    predicate_terms: List[str] = Field(default_factory=list)
    target_years: List[str] = Field(default_factory=list)


def extract_anchor_entities(question: str) -> List[str]:
    anchors = _ACRONYM_RE.findall(question)
    # Deduplicate preserving order
    seen = set()
    out = []
    for a in anchors:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def extract_predicate_terms(text: str, anchor_entities: List[str]) -> List[str]:
    anchor_low = {a.lower() for a in anchor_entities}
    toks = tokenize(text, remove_stopwords=True)
    out = []
    seen = set()
    for t in toks:
        if t in anchor_low or t in _BOILERPLATE_TERMS or len(t) <= 2 or t.isdigit():
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


class RuleBasedClaimDecomposer:
    """
    Deterministic syntactic claim decomposer.
    Splits multi-hop / multi-clause questions into atomic claims with explicit importance weights.
    """

    def decompose(self, question: str, enabled: bool = True) -> List[AtomicClaim]:
        anchors = extract_anchor_entities(question)
        years = sorted(set(_YEAR_RE.findall(question)))

        if not enabled:
            return [
                AtomicClaim(
                    claim_id="C1",
                    text=question.strip(),
                    weight=1.0,
                    claim_type="factual",
                    anchor_entities=anchors,
                    predicate_terms=extract_predicate_terms(question, anchors),
                    target_years=years,
                )
            ]

        q_clean = question.strip().rstrip("?").strip()
        q_low = q_clean.lower()
        claims: List[AtomicClaim] = []

        # Pattern 1: Delegated schedule / annex multi-hop question ("... and what ... delegated schedule")
        if re.search(r"\band\s+what\b.*\b(delegated|schedule|annex|protocol|supplement|addendum|bulletin)\b", q_low):
            parts = re.split(r"\band\s+what\b", q_clean, maxsplit=1, flags=re.IGNORECASE)
            left = parts[0].strip().rstrip(",")
            right = ("what " + parts[1].strip()) if len(parts) > 1 else ""
            claims.append(
                AtomicClaim(
                    claim_id="C1",
                    text=left,
                    weight=1.0,
                    claim_type="temporal_amendment",
                    anchor_entities=anchors,
                    predicate_terms=extract_predicate_terms(left, anchors),
                    target_years=years,
                )
            )
            claims.append(
                AtomicClaim(
                    claim_id="C2",
                    text=right,
                    weight=0.9,
                    claim_type="delegated_schedule",
                    anchor_entities=anchors,
                    predicate_terms=extract_predicate_terms(right, anchors),
                    target_years=years,
                )
            )
            return claims

        # Pattern 2: Joint multi-clause question ("which X and which Y ...")
        if re.search(r"\bwhich\b.+\band\s+which\b", q_low):
            # Preserve any leading context before the first "which"
            m = re.search(r"^(.*?)(\bwhich\b.*)$", q_clean, flags=re.IGNORECASE)
            prefix = m.group(1).strip() if m else ""
            body = m.group(2).strip() if m else q_clean
            subparts = re.split(r"\band\s+(?=which\b)", body, flags=re.IGNORECASE)
            for idx, sp in enumerate(subparts, start=1):
                sp_text = f"{prefix} {sp}".strip() if prefix else sp.strip()
                # Heuristic: second clause after an amendment often corresponds to the amended provision
                ctype = "temporal_amendment" if (idx == len(subparts) and "amendment" in q_low) else "factual"
                claims.append(
                    AtomicClaim(
                        claim_id=f"C{idx}",
                        text=sp_text,
                        weight=1.0,
                        claim_type=ctype,
                        anchor_entities=anchors,
                        predicate_terms=extract_predicate_terms(sp, anchors),
                        target_years=years,
                    )
                )
            return claims

        # Pattern 3: Conflict-check question ("... and is there an unresolved conflict ...")
        if re.search(r"\band\s+(is\s+there|are\s+there)\b.*\b(conflict|discrepancy|contradict)", q_low):
            parts = re.split(r",?\s+and\s+(?=is\s+there|are\s+there)", q_clean, maxsplit=1, flags=re.IGNORECASE)
            left = parts[0].strip()
            right = parts[1].strip() if len(parts) > 1 else "unresolved conflict between authorities"
            claims.append(
                AtomicClaim(
                    claim_id="C1",
                    text=left,
                    weight=1.0,
                    claim_type="factual",
                    anchor_entities=anchors,
                    predicate_terms=extract_predicate_terms(left, anchors),
                    target_years=years,
                )
            )
            claims.append(
                AtomicClaim(
                    claim_id="C2",
                    text=right,
                    weight=0.85,
                    claim_type="conflict_check",
                    anchor_entities=anchors,
                    predicate_terms=extract_predicate_terms(left, anchors),
                    target_years=years,
                )
            )
            return claims

        # Default: Single atomic claim
        ctype = "temporal_amendment" if any(k in q_low for k in ("currently binding", "revision", "amendment")) else "factual"
        claims.append(
            AtomicClaim(
                claim_id="C1",
                text=q_clean,
                weight=1.0,
                claim_type=ctype,
                anchor_entities=anchors,
                predicate_terms=extract_predicate_terms(q_clean, anchors),
                target_years=years,
            )
        )
        return claims
