"""Typed data schemas for ChronoEvidence documents, queries, runs, and provenance."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    doc_id: str
    family_id: str = Field(description="Logical policy/topic family linking document versions")
    version: str = Field(default="v1")
    domain: str
    title: str
    text: str
    publication_date: str = Field(description="ISO YYYY-MM-DD publication date")
    effective_from: str = Field(description="ISO YYYY-MM-DD effective start date")
    effective_to: Optional[str] = Field(
        default=None, description="ISO YYYY-MM-DD effective end date; None means currently in force"
    )
    clause_id: Optional[str] = None
    supersedes: List[str] = Field(default_factory=list)
    modifies: List[str] = Field(default_factory=list)
    conflicts_with: List[str] = Field(default_factory=list)


class QueryRecord(BaseModel):
    query_id: str
    split: str = Field(description="dev or test")
    category: str = Field(
        description=(
            "Failure-mode category: missing_hop, outdated_decoy, partial_amendment, "
            "pub_vs_effective_trap, unresolved_conflict, insufficient_evidence"
        )
    )
    domain: str
    question: str
    target_date: Optional[str] = Field(
        default=None, description="ISO YYYY-MM-DD date requested by query, if applicable"
    )
    required_periods: List[str] = Field(
        default_factory=list, description="Required temporal periods/versions (e.g., ['2022', '2025'])"
    )
    gold_doc_ids: List[str] = Field(
        default_factory=list, description="All gold evidence passages required to answer completely"
    )
    outdated_decoy_ids: List[str] = Field(
        default_factory=list, description="Known stale/superseded/premature decoys for error analysis"
    )
    requires_abstention: bool = False
    has_true_conflict: bool = False
    notes: str = ""


class RetrievedItem(BaseModel):
    doc_id: str
    rank: int
    score: float
    method: str
    component_ranks: Dict[str, int] = Field(default_factory=dict)
    component_scores: Dict[str, float] = Field(default_factory=dict)
