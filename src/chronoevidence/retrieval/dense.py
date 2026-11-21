"""Dense retrieval baselines: offline Dense-LSA (SVD + PyTorch cosine) and production SentenceTransformer + FAISS."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from chronoevidence.retrieval.base import BaseRetriever
from chronoevidence.schemas import DocumentRecord, RetrievedItem


class DenseLSARetriever(BaseRetriever):
    """
    Offline continuous dense vector baseline (Word+Char TF-IDF -> TruncatedSVD -> L2-normalized dense
    embeddings searched via exact PyTorch cosine similarity).

    Explicitly named `dense_lsa_svd` to maintain strict scientific honesty in air-gapped environments
    where pretrained neural encoders (e.g., BAAI/bge-m3) cannot be downloaded.
    """

    def __init__(self, embedding_dim: int = 96, seed: int = 42) -> None:
        self.name = "dense_lsa_svd"
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        self.svd: TruncatedSVD | None = None
        self.doc_ids: List[str] = []
        self.doc_embeddings: torch.Tensor | None = None

    def _encode_sparse(self, texts: Sequence[str], fit: bool) -> object:
        if fit:
            xw = self.word_vec.fit_transform(texts)
            xc = self.char_vec.fit_transform(texts)
        else:
            xw = self.word_vec.transform(texts)
            xc = self.char_vec.transform(texts)
        return hstack([xw, xc]).tocsr()

    def index(self, documents: Sequence[DocumentRecord]) -> None:
        self.doc_ids = [d.doc_id for d in documents]
        texts = [f"{d.title}. {d.text}" for d in documents]
        x_sparse = self._encode_sparse(texts, fit=True)
        n_components = min(self.embedding_dim, max(2, min(x_sparse.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        dense = self.svd.fit_transform(x_sparse)
        dense = normalize(dense, norm="l2", axis=1)
        self.doc_embeddings = torch.tensor(dense, dtype=torch.float32)

    def retrieve(self, query: str, top_k: int = 10, target_date: str | None = None) -> List[RetrievedItem]:
        if self.svd is None or self.doc_embeddings is None:
            raise RuntimeError("Call index() before retrieve().")
        q_sparse = self._encode_sparse([query], fit=False)
        q_dense = self.svd.transform(q_sparse)
        q_dense = normalize(q_dense, norm="l2", axis=1)
        q_tensor = torch.tensor(q_dense[0], dtype=torch.float32)

        with torch.no_grad():
            scores = torch.matmul(self.doc_embeddings, q_tensor).cpu().numpy()

        order = sorted(range(len(self.doc_ids)), key=lambda i: (-float(scores[i]), self.doc_ids[i]))[:top_k]
        return [
            RetrievedItem(
                doc_id=self.doc_ids[idx],
                rank=rank,
                score=float(scores[idx]),
                method=self.name,
                component_ranks={self.name: rank},
                component_scores={self.name: float(scores[idx])},
            )
            for rank, idx in enumerate(order, start=1)
        ]

    def get_metadata(self) -> Dict[str, object]:
        actual_dim = int(self.doc_embeddings.shape[1]) if self.doc_embeddings is not None else self.embedding_dim
        return {
            "retriever": self.name,
            "backend": "sklearn_tfidf_svd_torch_cosine",
            "embedding_dim": actual_dim,
            "seed": self.seed,
            "corpus_size": len(self.doc_ids),
        }


class SentenceTransformerFAISSRetriever(BaseRetriever):
    """Production neural dense retriever using a pinned SentenceTransformer model + FAISS IndexFlatIP."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        revision: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.name = f"dense_faiss_{model_name.split('/')[-1]}"
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.doc_ids: List[str] = []
        self._model = None
        self._index = None

    def index(self, documents: Sequence[DocumentRecord]) -> None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers and faiss-cpu are required for SentenceTransformerFAISSRetriever. "
                "Install requirements-full.txt on an internet-enabled environment."
            ) from exc

        self.doc_ids = [d.doc_id for d in documents]
        self._model = SentenceTransformer(self.model_name, revision=self.revision, device=self.device)
        texts = [f"{d.title}. {d.text}" for d in documents]
        embs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        self._index = faiss.IndexFlatIP(embs.shape[1])
        self._index.add(embs)

    def retrieve(self, query: str, top_k: int = 10, target_date: str | None = None) -> List[RetrievedItem]:
        if self._model is None or self._index is None:
            raise RuntimeError("Call index() before retrieve().")
        q_emb = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        scores, indices = self._index.search(q_emb, min(top_k, len(self.doc_ids)))
        results: List[RetrievedItem] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            results.append(
                RetrievedItem(
                    doc_id=self.doc_ids[int(idx)],
                    rank=rank,
                    score=float(score),
                    method=self.name,
                    component_ranks={self.name: rank},
                    component_scores={self.name: float(score)},
                )
            )
        return results
