"""Unit tests for Phase 4 ACEC controller covering all 7 required edge cases."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronoevidence.controller.acec import ACECConfig, ACECController
from chronoevidence.retrieval.bm25 import BM25OkapiRetriever
from chronoevidence.schemas import DocumentRecord


def make_edge_case_corpus():
    return [
        DocumentRecord(
            doc_id="pol_v1_base",
            family_id="pol",
            version="v1.0",
            domain="policy",
            title="Alpha Storage Code (ASC) - Original 2022 Framework",
            text=(
                "Under Clause A regarding perimeter concrete setback, operators must maintain 30 meters. "
                "Under Clause B regarding lithium rack capacity limit, the limit is strictly 5 MWh."
            ),
            publication_date="2022-03-01",
            effective_from="2022-04-01",
            effective_to=None,
            clause_id="Clause A, Clause B (v1)",
        ),
        DocumentRecord(
            doc_id="pol_v1_faq_old",
            family_id="pol",
            version="v1.1",
            domain="policy",
            title="ASC FAQ on Lithium Rack Capacity Limit (2023)",
            text=(
                "Frequently Asked Question on ASC lithium rack capacity limit: "
                "Operators must comply with 5 MWh for lithium rack capacity limit."
            ),
            publication_date="2023-06-01",
            effective_from="2023-06-01",
            effective_to="2025-02-28",
            clause_id="Clause B (Superseded)",
        ),
        DocumentRecord(
            doc_id="pol_v2_amend",
            family_id="pol",
            version="v2.0",
            domain="policy",
            title="ASC Amendment Order 2025-02 (Revision of Clause B)",
            text=(
                "Pursuant to statutory review of ASC, Clause B is hereby revised effective March 1, 2025: "
                "the revised ceiling is tightened to 2.5 MWh. Clause A remains unmodified. "
                "Technical verification mechanics are delegated to Engineering Supplement ES-9 (Thermal Venting)."
            ),
            publication_date="2025-02-01",
            effective_from="2025-03-01",
            effective_to=None,
            clause_id="Clause B (v2)",
            supersedes=["pol_v1_faq_old"],
            modifies=["pol_v1_base"],
        ),
        DocumentRecord(
            doc_id="pol_annex",
            family_id="pol",
            version="v2.0-annex",
            domain="policy",
            title="Engineering Supplement ES-9 (Thermal Venting)",
            text=(
                "Engineering Supplement ES-9 (Thermal Venting) defines mandatory implementation parameters "
                "referenced by ASC Amendment Order 2025-02. Compliance engineering teams must verify gas "
                "scrubbers trigger at 25 ppm."
            ),
            publication_date="2025-02-15",
            effective_from="2025-03-01",
            effective_to=None,
            clause_id="Annex",
        ),
        DocumentRecord(
            doc_id="pol_clause_c_active",
            family_id="pol",
            version="v1.2",
            domain="policy",
            title="ASC Clause C Sensor Telemetry Rule (2023-2026)",
            text=(
                "Under ASC Clause C governing thermal sensor sampling, the active requirement through "
                "June 30, 2026 mandates sampling once every 60 seconds."
            ),
            publication_date="2022-12-01",
            effective_from="2023-01-01",
            effective_to="2026-06-30",
            clause_id="Clause C (v1)",
        ),
        DocumentRecord(
            doc_id="pol_clause_c_future",
            family_id="pol",
            version="v2.1",
            domain="policy",
            title="ASC 2026 Modernization Notice: Thermal Sensor Sampling",
            text=(
                "Published November 15, 2025: ASC updates thermal sensor sampling to 10 Hz. "
                "Does NOT take legal effect until July 1, 2026."
            ),
            publication_date="2025-11-15",
            effective_from="2026-07-01",
            effective_to=None,
            clause_id="Clause C (v2 - Future)",
            supersedes=["pol_clause_c_active"],
        ),
        DocumentRecord(
            doc_id="pol_conf_a",
            family_id="pol",
            version="v1-authA",
            domain="policy",
            title="ASC Sector Directive A on Fire Suppression Agent",
            text="Binding Sector Directive A for ASC addressing fire suppression agent: must use water deluge.",
            publication_date="2025-04-20",
            effective_from="2025-05-01",
            effective_to=None,
            conflicts_with=["pol_conf_b"],
        ),
        DocumentRecord(
            doc_id="pol_conf_b",
            family_id="pol",
            version="v1-authB",
            domain="policy",
            title="ASC Parallel Order B on Fire Suppression Agent",
            text="Parallel Regulatory Order B for ASC governing fire suppression agent: must use dry aerosol gas.",
            publication_date="2025-05-25",
            effective_from="2025-06-01",
            effective_to=None,
            conflicts_with=["pol_conf_a"],
        ),
    ]


class TestACECEdgeCases(unittest.TestCase):
    def setUp(self):
        self.docs = make_edge_case_corpus()
        self.bm25 = BM25OkapiRetriever()
        self.ctrl = ACECController(
            base_retriever=self.bm25,
            config=ACECConfig(budget_calls=3, per_call_depth=4),
        )
        self.ctrl.index(self.docs)

    def test_1_missing_hop_delegated_schedule(self):
        res = self.ctrl.run(
            "Under the 2025 ASC revision to lithium rack capacity limit, what is the updated limit "
            "and what exact technical verification parameters are required by its delegated schedule?",
            top_k=5,
            target_date="2025-06-01",
        )
        top_ids = [r.doc_id for r in res.retrieved[:3]]
        self.assertIn("pol_v2_amend", top_ids)
        self.assertIn("pol_annex", top_ids)
        self.assertFalse(res.abstain)
        self.assertGreaterEqual(len(res.claims), 2)

    def test_2_outdated_version_superseded(self):
        res = self.ctrl.run(
            "What is the currently binding lithium rack capacity limit requirement under ASC as of January 2026?",
            top_k=5,
            target_date="2026-01-15",
        )
        self.assertEqual(res.retrieved[0].doc_id, "pol_v2_amend")
        self.assertNotEqual(res.retrieved[0].doc_id, "pol_v1_faq_old")
        self.assertFalse(res.abstain)

    def test_3_partial_amendment_preserves_clause_a(self):
        res = self.ctrl.run(
            "Following the 2025 ASC amendment, which perimeter concrete setback rule and which "
            "lithium rack capacity limit ceiling jointly apply in 2026?",
            top_k=5,
            target_date="2026-04-01",
        )
        top2 = {r.doc_id for r in res.retrieved[:2]}
        self.assertEqual(top2, {"pol_v1_base", "pol_v2_amend"})
        self.assertFalse(res.abstain)

    def test_4_contradictory_effective_date_trap(self):
        res = self.ctrl.run(
            "On March 15, 2026, what is the legally enforceable ASC rule for thermal sensor sampling?",
            top_k=5,
            target_date="2026-03-15",
        )
        self.assertEqual(res.retrieved[0].doc_id, "pol_clause_c_active")
        self.assertFalse(res.abstain)

    def test_5_conflicting_documents_detected(self):
        res = self.ctrl.run(
            "What regulatory requirement governs fire suppression agent under ASC in 2026, "
            "and is there an unresolved conflict between authorities?",
            top_k=5,
            target_date="2026-05-01",
        )
        top2 = {r.doc_id for r in res.retrieved[:2]}
        self.assertEqual(top2, {"pol_conf_a", "pol_conf_b"})
        self.assertTrue(res.has_conflict)
        self.assertFalse(res.abstain)

    def test_6_insufficient_evidence_triggers_abstention(self):
        res = self.ctrl.run(
            "Under the 2026 Alpha Storage Code (ASC), what exact statutory limit governs submarine sonar whale mitigation?",
            top_k=5,
            target_date="2026-05-01",
        )
        self.assertTrue(res.abstain)
        self.assertLess(res.C_q, 0.45)

    def test_7_budget_exhaustion_enforced(self):
        tight_ctrl = ACECController(
            base_retriever=self.bm25,
            config=ACECConfig(budget_calls=1, per_call_depth=2),
        )
        tight_ctrl.index(self.docs, index_base=False)
        res = tight_ctrl.run(
            "Under the 2026 Alpha Storage Code (ASC), what exact statutory limit governs submarine sonar whale mitigation?",
            top_k=5,
            target_date="2026-05-01",
        )
        self.assertEqual(res.retrieval_calls, 1)
        self.assertEqual(res.stop_reason, "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
