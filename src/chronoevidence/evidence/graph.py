"""NetworkX provenance and temporal evidence graph for ACEC."""

from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from chronoevidence.schemas import DocumentRecord

# Extract explicit cross-document delegation pointers from retrieved document text
_DELEGATION_RE = re.compile(
    r"delegated\s+to\s+([A-Z][A-Za-z0-9\-\s]+?\([A-Za-z0-9\-\s]+?\))",
    flags=re.IGNORECASE,
)


def extract_delegation_targets(text: str) -> List[str]:
    return [m.group(1).strip() for m in _DELEGATION_RE.finditer(text)]


class EvidenceGraph:
    """
    Directed provenance graph connecting retrieved DocumentRecords via:
      - SUPERSEDES (newer doc fully replaces older doc)
      - MODIFIES   (newer doc partially amends older doc; unmodified clauses stay active)
      - CONFLICTS_WITH (concurrently active contradictory directives)
      - REFERS_TO  (text-extracted delegation pointer to an annex/schedule)
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.docs: Dict[str, DocumentRecord] = {}
        self.delegation_pointers: Dict[str, List[str]] = {}

    def add_document(self, doc: DocumentRecord) -> None:
        self.docs[doc.doc_id] = doc
        self.graph.add_node(
            doc.doc_id,
            node_type="Document",
            family_id=doc.family_id,
            version=doc.version,
            publication_date=doc.publication_date,
            effective_from=doc.effective_from,
            effective_to=doc.effective_to,
            clause_id=doc.clause_id,
        )
        for target_id in doc.supersedes:
            self.graph.add_edge(doc.doc_id, target_id, relation="SUPERSEDES")
        for target_id in doc.modifies:
            self.graph.add_edge(doc.doc_id, target_id, relation="MODIFIES")
        for target_id in doc.conflicts_with:
            self.graph.add_edge(doc.doc_id, target_id, relation="CONFLICTS_WITH")

        refs = extract_delegation_targets(doc.text)
        if refs:
            self.delegation_pointers[doc.doc_id] = refs
            for ref in refs:
                ref_node = f"REF::{ref}"
                self.graph.add_node(ref_node, node_type="ExternalReference", title=ref)
                self.graph.add_edge(doc.doc_id, ref_node, relation="REFERS_TO")

    def is_temporally_valid(self, doc_id: str, target_date: Optional[str]) -> bool:
        if not target_date or doc_id not in self.docs:
            return True
        doc = self.docs[doc_id]
        t_dt = date.fromisoformat(target_date[:10])
        start_dt = date.fromisoformat(doc.effective_from[:10])
        if t_dt < start_dt:
            return False
        if doc.effective_to:
            end_dt = date.fromisoformat(doc.effective_to[:10])
            if t_dt > end_dt:
                return False
        return True

    def is_superseded_in_graph(self, doc_id: str, target_date: Optional[str]) -> bool:
        """True if an active retrieved document explicitly SUPERSEDES doc_id as of target_date."""
        if doc_id not in self.graph:
            return False
        for pred in self.graph.predecessors(doc_id):
            edge_data = self.graph.get_edge_data(pred, doc_id)
            if edge_data and edge_data.get("relation") == "SUPERSEDES":
                if pred in self.docs and self.is_temporally_valid(pred, target_date):
                    return True
        return False

    def get_active_modifiers_of(self, doc_id: str, target_date: Optional[str]) -> List[str]:
        """Return retrieved active documents that partially MODIFY doc_id."""
        if doc_id not in self.graph:
            return []
        mods = []
        for pred in self.graph.predecessors(doc_id):
            edge_data = self.graph.get_edge_data(pred, doc_id)
            if edge_data and edge_data.get("relation") == "MODIFIES":
                if pred in self.docs and self.is_temporally_valid(pred, target_date):
                    mods.append(pred)
        return mods

    def get_unresolved_conflicts(self, target_date: Optional[str]) -> List[Tuple[str, str]]:
        """Return active document pairs linked by CONFLICTS_WITH."""
        pairs: Set[Tuple[str, str]] = set()
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "CONFLICTS_WITH":
                if u in self.docs and self.is_temporally_valid(u, target_date):
                    if v in self.docs and self.is_temporally_valid(v, target_date):
                        pairs.add(tuple(sorted((u, v))))
        return sorted(pairs)

    def get_missing_conflict_partners(self, target_date: Optional[str]) -> List[Tuple[str, str]]:
        """Return (retrieved_doc_id, missing_partner_id) where CONFLICTS_WITH target is not yet retrieved."""
        missing = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "CONFLICTS_WITH":
                if u in self.docs and self.is_temporally_valid(u, target_date) and v not in self.docs:
                    missing.append((u, v))
        return missing

    def get_unfetched_delegations(self) -> List[Tuple[str, str]]:
        """Return (source_doc_id, ref_title) for delegation pointers not yet matched by a retrieved doc title."""
        fetched_titles = {d.title.lower() for d in self.docs.values()}
        unfetched = []
        for src_id, refs in self.delegation_pointers.items():
            for ref in refs:
                if not any(ref.lower() in t or t in ref.lower() for t in fetched_titles):
                    unfetched.append((src_id, ref))
        return unfetched

    def summary(self) -> Dict[str, int]:
        rel_counts: Dict[str, int] = {}
        for _, _, d in self.graph.edges(data=True):
            rel = d.get("relation", "UNKNOWN")
            rel_counts[rel] = rel_counts.get(rel, 0) + 1
        return {
            "num_docs": len(self.docs),
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            **rel_counts,
        }
