"""Abstract retriever interface for ChronoEvidence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

from chronoevidence.schemas import DocumentRecord, RetrievedItem


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def index(self, documents: Sequence[DocumentRecord]) -> None:
        """Build or load an index over the corpus."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10, target_date: str | None = None) -> List[RetrievedItem]:
        """Return top-k ranked documents for a query."""

    def get_metadata(self) -> Dict[str, object]:
        return {"retriever": self.name}
