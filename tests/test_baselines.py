"""Unit tests for Phase 3 retrieval baselines, evaluation metrics, determinism, and split integrity."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronoevidence.evaluation.metrics import (
    decoy_intrusion_rate_at_k,
    evidence_completeness_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    temporal_coverage_at_k,
    temporal_precision_at_k,
)
from chronoevidence.retrieval.bm25 import BM25OkapiRetriever
from chronoevidence.retrieval.dense import DenseLSARetriever
from chronoevidence.retrieval.hybrid import HybridRRFRetriever
from chronoevidence.schemas import DocumentRecord, QueryRecord
from chronoevidence.temporal.recency import TemporalRecencyRetriever


def make_toy_docs():
    return [
        DocumentRecord(
            doc_id="d1_old",
            family_id="fam1",
            version="v1",
            domain="policy",
            title="Battery Capacity Rule 2022",
            text="Maximum lithium storage container limit is 5 megawatt hours under 2022 rule.",
            publication_date="2022-03-01",
            effective_from="2022-04-01",
            effective_to="2025-02-28",
        ),
        DocumentRecord(
            doc_id="d2_new",
            family_id="fam1",
            version="v2",
            domain="policy",
            title="Battery Capacity Amendment 2025",
            text="Revised lithium storage enclosure ceiling is 2.5 megawatt hours effective March 2025.",
            publication_date="2025-02-01",
            effective_from="2025-03-01",
            effective_to=None,
        ),
        DocumentRecord(
            doc_id="d3_irrel",
            family_id="fam2",
            version="v1",
            domain="aviation",
            title="Runway De-Icing Standard",
            text="Aircraft runway glycol runoff permits require seasonal drainage sampling.",
            publication_date="2026-01-01",
            effective_from="2026-01-01",
            effective_to=None,
        ),
    ]


class TestRetrievalAndMetrics(unittest.TestCase):
    def test_bm25_ranks_relevant_above_irrelevant_and_is_deterministic(self):
        docs = make_toy_docs()
        r1 = BM25OkapiRetriever(k1=1.5, b=0.75)
        r2 = BM25OkapiRetriever(k1=1.5, b=0.75)
        r1.index(docs)
        r2.index(docs)
        h1 = r1.retrieve("lithium storage megawatt hours", top_k=3)
        h2 = r2.retrieve("lithium storage megawatt hours", top_k=3)
        self.assertEqual([x.doc_id for x in h1], [x.doc_id for x in h2])
        self.assertEqual(h1[-1].doc_id, "d3_irrel")
        self.assertGreater(h1[0].score, h1[-1].score)

    def test_dense_lsa_determinism_and_cosine_bounds(self):
        docs = make_toy_docs()
        d1 = DenseLSARetriever(embedding_dim=2, seed=42)
        d2 = DenseLSARetriever(embedding_dim=2, seed=42)
        d1.index(docs)
        d2.index(docs)
        h1 = d1.retrieve("lithium storage ceiling", top_k=3)
        h2 = d2.retrieve("lithium storage ceiling", top_k=3)
        self.assertEqual([x.doc_id for x in h1], [x.doc_id for x in h2])
        for item in h1:
            self.assertGreaterEqual(item.score, -1.0001)
            self.assertLessEqual(item.score, 1.0001)

    def test_hybrid_rrf_exact_formula(self):
        docs = make_toy_docs()
        bm25 = BM25OkapiRetriever()
        dense = DenseLSARetriever(embedding_dim=2, seed=42)
        hybrid = HybridRRFRetriever([bm25, dense], weights=[1.0, 1.0], rrf_k=60, candidate_depth=3)
        hybrid.index(docs)
        hits = hybrid.retrieve("lithium storage", top_k=3)
        top = hits[0]
        expected = sum(1.0 / (60 + r) for r in top.component_ranks.values())
        self.assertAlmostEqual(top.score, expected, places=7)

    def test_temporal_recency_boosts_newer_same_topic_doc(self):
        docs = make_toy_docs()
        bm25 = BM25OkapiRetriever()
        temp = TemporalRecencyRetriever(bm25, alpha=0.5, decay_lambda=0.5, reference_date="2026-01-01")
        temp.index(docs)
        hits = temp.retrieve("lithium storage", top_k=3, target_date="2026-01-01")
        self.assertEqual(hits[0].doc_id, "d2_new")

    def test_metrics_exact_math(self):
        retrieved = ["x1", "g1", "x2", "g2", "x3"]
        gold = ["g1", "g2"]
        self.assertAlmostEqual(recall_at_k(retrieved, gold, 1), 0.0)
        self.assertAlmostEqual(recall_at_k(retrieved, gold, 3), 0.5)
        self.assertAlmostEqual(recall_at_k(retrieved, gold, 5), 1.0)
        self.assertAlmostEqual(hit_rate_at_k(retrieved, gold, 1), 0.0)
        self.assertAlmostEqual(hit_rate_at_k(retrieved, gold, 5), 1.0)
        self.assertAlmostEqual(mrr_at_k(retrieved, gold, 5), 0.5)
        self.assertAlmostEqual(evidence_completeness_at_k(retrieved, gold, 3), 0.0)
        self.assertAlmostEqual(evidence_completeness_at_k(retrieved, gold, 5), 1.0)
        expected_dcg = 1.0 / math.log2(2 + 1) + 1.0 / math.log2(4 + 1)
        expected_idcg = 1.0 / math.log2(1 + 1) + 1.0 / math.log2(2 + 1)
        self.assertAlmostEqual(ndcg_at_k(retrieved, gold, 5), expected_dcg / expected_idcg, places=7)
        self.assertAlmostEqual(decoy_intrusion_rate_at_k(retrieved, ["x1"], 3), 1.0)
        self.assertAlmostEqual(decoy_intrusion_rate_at_k(retrieved, ["x9"], 3), 0.0)

    def test_temporal_metrics(self):
        docs = make_toy_docs()
        doc_map = {d.doc_id: d for d in docs}
        q = QueryRecord(
            query_id="q1",
            split="dev",
            category="partial_amendment",
            domain="policy",
            question="Compare 2022 and 2025",
            target_date="2026-01-15",
            required_periods=["2022", "2025"],
            gold_doc_ids=["d1_old", "d2_new"],
        )
        # On 2026-01-15: d1_old expired 2025-02-28 (invalid), d2_new valid, d3_irrel valid
        self.assertAlmostEqual(temporal_precision_at_k(["d1_old", "d2_new"], q, doc_map, 2), 0.5)
        self.assertAlmostEqual(temporal_coverage_at_k(["d1_old", "d2_new"], q, doc_map, 2), 1.0)
        self.assertAlmostEqual(temporal_coverage_at_k(["d2_new", "d3_irrel"], q, doc_map, 2), 0.5)

    def test_chronodiag60_split_integrity(self):
        ds_dir = REPO_ROOT / "data" / "processed" / "chronodiag60"
        if not (ds_dir / "queries_dev.jsonl").exists():
            self.skipTest("ChronoDiag-60 not built yet")
        dev = [json.loads(l) for l in (ds_dir / "queries_dev.jsonl").read_text().splitlines() if l.strip()]
        test = [json.loads(l) for l in (ds_dir / "queries_test.jsonl").read_text().splitlines() if l.strip()]
        dev_ids = {q["query_id"] for q in dev}
        test_ids = {q["query_id"] for q in test}
        self.assertEqual(len(dev_ids & test_ids), 0, "Leakage between dev and test query IDs!")
        # Verify zero gold_doc_id overlap between dev and test (strict topic-family separation)
        dev_golds = {g for q in dev for g in q["gold_doc_ids"]}
        test_golds = {g for q in test for g in q["gold_doc_ids"]}
        self.assertEqual(len(dev_golds & test_golds), 0, "Gold document overlap between dev and test!")


if __name__ == "__main__":
    unittest.main()
