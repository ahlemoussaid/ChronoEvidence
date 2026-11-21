"""Deterministic extractive grounded generator under a fixed context-token budget."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from pydantic import BaseModel

from chronoevidence.schemas import DocumentRecord, RetrievedItem


class GroundedAnswer(BaseModel):
    abstained: bool
    answer_text: str
    cited_doc_ids: List[str]
    context_tokens: int
    flagged_conflict: bool


def count_tokens(text: str) -> int:
    return len(text.split())


class GroundedExtractiveGenerator:
    """
    Budget-matched extractive grounded generator.
    Accepts at most `max_passages` and `max_context_tokens` from ANY retriever (baseline or ACEC),
    producing claim-level citations or abstaining when instructed.
    """

    def __init__(self, max_passages: int = 5, max_context_tokens: int = 450) -> None:
        self.max_passages = max_passages
        self.max_context_tokens = max_context_tokens

    def generate(
        self,
        retrieved: Sequence[RetrievedItem],
        doc_map: Dict[str, DocumentRecord],
        abstain: bool = False,
        explicit_citations: Optional[List[str]] = None,
        flagged_conflict: bool = False,
    ) -> GroundedAnswer:
        selected_docs: List[DocumentRecord] = []
        used_tokens = 0
        for item in retrieved[: self.max_passages]:
            doc = doc_map.get(item.doc_id)
            if doc is None:
                continue
            toks = count_tokens(f"{doc.title} {doc.text}")
            if used_tokens + toks > self.max_context_tokens and selected_docs:
                break
            selected_docs.append(doc)
            used_tokens += toks

        if abstain:
            return GroundedAnswer(
                abstained=True,
                answer_text="ABSTAIN: Insufficient temporally valid evidence to answer this query.",
                cited_doc_ids=[],
                context_tokens=used_tokens,
                flagged_conflict=flagged_conflict,
            )

        selected_ids = [d.doc_id for d in selected_docs]
        if explicit_citations is not None:
            cited = [d for d in dict.fromkeys(explicit_citations) if d in selected_ids]
            if not cited and selected_ids:
                cited = selected_ids[:1]
        else:
            # Conventional baseline generator cites its top-2 context passages under the same budget
            cited = selected_ids[:2]

        snippets = [f"[{d.doc_id}] {d.title}: {d.text[:160]}" for d in selected_docs if d.doc_id in cited]
        prefix = "[CONFLICT DETECTED] " if flagged_conflict else ""
        return GroundedAnswer(
            abstained=False,
            answer_text=prefix + " | ".join(snippets),
            cited_doc_ids=cited,
            context_tokens=used_tokens,
            flagged_conflict=flagged_conflict,
        )
