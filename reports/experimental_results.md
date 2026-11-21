# ChronoEvidence — Phase 5 Controlled Experimental Results

**Date:** 2026-11-20  
**Experiment ID:** `phase5_acec_vs_baselines_and_ablations`  
**Config:** `configs/acec_chronodiag60.json` | **Seed:** `42` | **Corpus:** `ChronoDiag-60` v1.0.0 (`150` docs)  
**Splits:** Dev (`N = 18`, `15` answerable, `3` unanswerable) — used for configuration lock; Held-Out Test (`N = 42`, `35` answerable, `7` unanswerable) — reported below.

> **Scope Honesty Notice:** Results below were executed offline on `ChronoDiag-60` with `dense_lsa_svd` (96D) and a deterministic extractive grounded generator (`max_passages = 5`, `max_context_tokens = 450`) due to air-gapped sandbox constraints. Full neural (`BAAI/bge-m3` + `FAISS`) and public benchmark (`TEMPO`, `TimelyQABench`) runs remain untested until executed on an internet/GPU host.

---

## 1. Controlled Comparison Under Matched Generation & Varied Retrieval Budgets (Test Split, `N = 42`)

All systems share the identical corpus (`sha256: 450872bb...0128cb0`), identical `top_k = 10` ranking evaluation, and identical grounded generation context budget (`max_passages = 5`, `max_context_tokens = 450`).

| Method | Calls | Ctx Tok | Latency (ms) | Recall@5 | MRR@10 | nDCG@10 | Comp.@5 | TempPrec@5 | TempCov@5 | Decoy@3 (↓) | Cit.Prec | Cit.Rec | Joint Ground. | Abst.Acc | False Abst. (↓) | Unsupp. Ans. (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bm25_okapi` | 1.00 | 340.3 | **4.219** | 0.5000 | 0.5295 | 0.5922 | 0.4000 | 0.3857 | 0.7000 | 0.7857 | 0.4000 | 0.5000 | 0.3611 | 0.8333 | **0.0000** | 1.0000 |
| `dense_lsa_svd` | 1.00 | 348.0 | 6.995 | 0.5000 | 0.6035 | 0.6203 | 0.4000 | 0.3476 | 0.7000 | 0.7381 | 0.4000 | 0.5000 | 0.3611 | 0.8333 | **0.0000** | 1.0000 |
| `hybrid_rrf` | 1.00 | 342.2 | 9.480 | 0.5000 | 0.5314 | 0.5883 | 0.4000 | 0.3810 | 0.7000 | 0.7381 | 0.4000 | 0.5000 | 0.3611 | 0.8333 | **0.0000** | 1.0000 |
| `temporal_recency_hybrid` | 1.00 | **313.9** | 10.809 | 0.5143 | 0.3852 | 0.4803 | 0.4000 | 0.5810 | 0.7857 | **0.3333** | 0.1571 | 0.1571 | 0.1310 | 0.8333 | **0.0000** | 1.0000 |
| `acec_budget_B1` (matched 1-call) | 1.00 | 337.7 | 12.200 | 0.5000 | 0.6000 | 0.5226 | 0.4000 | 0.4905 | 0.7000 | 0.7381 | 0.6000 | 0.5000 | 0.5873 | 0.8095 | 0.2000 | 0.1429 |
| `acec_budget_B2` | 1.67 | 344.5 | 18.161 | 0.7000 | **0.8000** | 0.7226 | 0.6000 | 0.7571 | 0.8571 | 0.4048 | **0.8000** | **0.7000** | **0.7778** | **1.0000** | **0.0000** | **0.0000** |
| **`acec_full` (`B = 3`)** | 2.17 | 339.3 | 22.969 | **0.8000** | **0.8000** | **0.8000** | **0.8000** | **0.7810** | **1.0000** | 0.4048 | 0.6000 | 0.6000 | 0.6667 | 0.8333 | 0.2000 | **0.0000** |
| `acec_budget_B4` | 2.67 | 335.5 | 29.480 | **0.8000** | **0.8000** | **0.8000** | **0.8000** | **0.7810** | **1.0000** | 0.4048 | 0.6000 | 0.6000 | 0.6667 | 0.8333 | 0.2000 | **0.0000** |

---

## 2. Paired Statistical Comparison: `acec_full` (`B=3`) vs. `hybrid_rrf` (Test Split)

| Metric | `acec_full` | `hybrid_rrf` | Delta (`ACEC - Hyb`) | 95% Bootstrap CI | Wilcoxon `p` | Wins / Ties / Regressions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Recall@5` (↑) | 0.8000 | 0.5000 | **+0.3000** | `[+0.1714, +0.4286]` | `0.00070` | **14** / 21 / 0 |
| `MRR@10` (↑) | 0.8000 | 0.5314 | **+0.2686** | `[+0.1535, +0.3898]` | `0.00061` | **14** / 20 / **1** |
| `nDCG@10` (↑) | 0.8000 | 0.5883 | **+0.2117** | `[+0.1161, +0.3118]` | `0.00076` | **14** / 20 / **1** |
| `Completeness@5` (↑) | 0.8000 | 0.4000 | **+0.4000** | `[+0.2564, +0.5714]` | `0.00018` | **14** / 21 / 0 |
| `TempPrec@5` (↑) | 0.7810 | 0.3810 | **+0.4000** | `[+0.3143, +0.4857]` | `<0.00001` | **35** / 7 / 0 |
| `TempCov@5` (↑) | 1.0000 | 0.7000 | **+0.3000** | `[+0.1714, +0.4286]` | `0.00070` | **14** / 21 / 0 |
| `DecoyIntrusion@3` (↓) | 0.4048 | 0.7381 | **-0.3333** | `[-0.5238, -0.1429]` | `0.00284` | **18** / 20 / **4** |
| `Citation Precision` (↑) | 0.6000 | 0.4000 | **+0.2000** | `[+0.1286, +0.2857]` | `0.00018` | **14** / 21 / 0 |
| `Citation Recall` (↑) | 0.6000 | 0.5000 | **+0.1000** | `[+0.0429, +0.1714]` | `0.00815` | **7** / 28 / 0 |
| `Joint Grounding Score` (↑) | 0.6667 | 0.3611 | **+0.3056** | `[+0.2024, +0.4167]` | `0.00005` | **21** / 21 / 0 |
| `Abstention Accuracy` (↑) | 0.8333 | 0.8333 | `0.0000` | `[-0.1667, +0.1667]` | `1.00000` | 7 / 28 / **7** |
| `Context Tokens` (↓) | 339.3 | 342.2 | `-2.98` | `[-6.43, +0.64]` | `0.14311` | 21 / 5 / 16 |
| `Retrieval Calls` (↓) | 2.17 | 1.00 | **+1.17** | `[+0.90, +1.43]` | `<0.00001` | 0 / 14 / **28** |
| `Latency (ms)` (↓) | 22.97 | 9.48 | **+13.49** | `[+10.47, +16.55]` | `<0.00001` | 4 / 0 / **38** |

---

## 3. Improvements vs. Regressions (Reported Separately)

### A. Verified Improvements
1. **Multi-Hop Evidence Completeness Doubled (`0.4000 -> 0.8000`, `p = 0.00018`):** On `partial_amendment` (7/7 Test queries) and `missing_hop` (7/7 Test queries), `acec_full` raises `Completeness@5` from `0.0000` (all non-temporal baselines) to **`1.0000`**.
2. **Temporal Precision & Coverage Substantially Improved:** `TempPrec@5` increases from `0.3810` (`hybrid_rrf`) and `0.5810` (`temporal_recency_hybrid`) to **`0.7810`** (`p < 0.00001`), while `TempCov@5` reaches **`1.0000`** (`p = 0.00070`). Crucially, unlike `temporal_recency_hybrid`, ACEC achieves `1.0000` `Recall@5` on `pub_vs_effective_trap` without inverting active vs. future-effective clauses.
3. **Zero Unsupported Answers on Unanswerable Queries (`1.0000 -> 0.0000`):** Every baseline answers 7/7 unanswerable `insufficient_evidence` queries (`unsupported_answer_rate = 1.0000`); `acec_full` abstains cleanly on 7/7 (`0.0000`).
4. **Matched Generation Context Budget:** `acec_full` uses slightly *fewer* context tokens (`339.3` vs `342.2` Hybrid RRF, `348.0` Dense-LSA) while raising `Joint Grounding Score` by **`+0.3056`** (`0.6667` vs `0.3611`, `p = 0.00005`).

### B. Verified Regressions & Negative Results (Do Not Hide)
1. **Regression 1 — Self-Inflicted False Abstention on 7/7 `missing_hop` Queries (`false_abstention_rate = 0.2000`):**
   - **Observed:** On `CD60-04-MH` through `CD60-10-MH`, `acec_full` retrieves BOTH gold documents at ranks `#1` and `#2` (`Recall@5 = 1.0000`, `Completeness@5 = 1.0000`, `C_q = 0.9763`, `T_q = 1.0000`), yet **abstains** (`abstained = True`), collapsing `Joint Grounding Score` on `missing_hop` to `0.0000` and causing `acec_full` (`0.6667`) to **underperform `acec_budget_B2` (`0.7778`)**!
   - **Exact Root Cause from Raw Log (`test_acec_full.jsonl`):** At Step 3, `U_q = 0.60*(1 - 0.9763) + 0.25 + 0.35 = 0.6142 > 0.55`. Two over-broad graph heuristics triggered simultaneously: (i) `graph.get_unfetched_delegations()` counted unfetched delegation pointers on *other-family* amendments pulled in at Step 2 (`cloud_sla_v2_2025_amend_b`, `med_privacy_v2_2025_amend_b`), adding `+0.25`; and (ii) the staleness heuristic misclassified the 2025 annex (`ai_audit_annex_bridge`, which has empty `modifies`/`supersedes`) as an unmodified stale base doc, adding `+0.35`.
2. **Regression 2 — Complete Failure on `outdated_decoy` (`Recall@5 = 0.0000`, 1 `nDCG@10` Regression vs. `hybrid_rrf`):**
   - **Observed:** On `CD60-04-OD` through `CD60-10-OD`, `acec_full` achieves `Recall@5 = 0.0000` and `nDCG@10 = 0.0000`, regressing vs. `hybrid_rrf` (`0.0413`, regression on `CD60-10-OD`) and `temporal_recency_hybrid` (`0.1795`, 4 paired `nDCG@10` regressions vs. temporal recency).
   - **Exact Root Cause from Raw Log (`CD60-04-OD`, `CD60-10-OD`):** Step 2 appended all 12 verbose predicate + full-act-title words to the amendment sub-query (`"HFMEA Amendment Order 2025 Revision Clause B ... compute threshold triggering mandatory pre-deployment safety audit high-risk foundation model evaluation act"`). This lexical dilution caused BM25/Dense to return the exact same 6 verbose 2021–2024 documents (`new_docs_added = 0`), wasting 2 retrieval calls without ever inspecting `v2_2025_amend_b`.
3. **Regression 3 — Matched Single-Call Budget (`acec_budget_B1`) Regresses on `nDCG@10`:**
   - **Observed:** At strictly matched `mean_calls = 1.00`, `acec_budget_B1` drops `nDCG@10` to `0.5226` vs. `0.6203` (`dense_lsa_svd`) and `0.5883` (`hybrid_rrf`), because `per_call_depth = 6` truncates rank 7–10 hits without iterative recovery.
4. **Regression 4 — Compute & Latency Overhead:**
   - **Observed:** `acec_full` requires `2.17` calls (`+1.17`, 28/42 queries use >1 call) and `22.97 ms/query` (`+13.49 ms` vs. `hybrid_rrf`, `+18.75 ms` vs. `bm25_okapi`).
5. **Regression 5 — Decoy Intrusion on Unanswerable Queries & vs. Temporal Recency:**
   - **Observed:** `acec_full` has 4 paired `decoy_intrusion@3` regressions vs. `hybrid_rrf` on `insufficient_evidence` (`CD60-04-IE`, `05-IE`, `06-IE`, `08-IE`) because valid base/press docs outranked expired 2021 drafts before ACEC abstained; and `temporal_recency_hybrid` retains slightly lower aggregate `Decoy@3` (`0.3333` vs. `0.4048`, `p = 0.549`, non-significant).
