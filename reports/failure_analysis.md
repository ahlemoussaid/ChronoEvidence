# ChronoEvidence — Phase 3 Empirical Failure Analysis (Generated Directly from Test Logs)

**Date:** 2026-11-20  
**Source Logs:** `results/raw/test_{bm25_okapi,dense_lsa_svd,hybrid_rrf,temporal_recency_hybrid}.jsonl`  
**Audit Guarantee:** Every query ID, document ID, rank, score, and date below was extracted programmatically from the saved Test split JSONL logs (`N = 42`; `35` answerable, `7` unanswerable).

---

## 1. Failure Taxonomy Across Baselines (Test Split)

| Failure Mode | Affected Baselines | Observed Test Metric Signature | Root Cause |
| :--- | :--- | :--- | :--- |
| 1. Evolving Semantic Conflict (Outdated Verbose Decoy) | BM25, Dense-LSA, Hybrid RRF | `outdated_decoy`: `Recall@5 = 0.0000`, `MRR@10 = 0.0143`, `Decoy@3 = 1.0000` | Superseded verbose FAQs outscore terse legal amendments. |
| 2. Missing Second-Hop Bridge Evidence | BM25, Dense-LSA, Hybrid RRF | `missing_hop`: `Completeness@5 = 0.0000`, `Recall@5 = 0.0000` | Single-pass retrieval cannot follow an indirect annex reference (`MP-7`) absent from the question. |
| 3. Partial Amendment Clause Erosion | All 4 baselines | `partial_amendment`: `Completeness@5 = 0.0000` (all 4 methods) | Non-temporal baselines miss the terse 2025 amendment; recency boosting demotes the active 2022 Clause A and injects cross-domain noise. |
| 4. Publication-Date vs. Effective-Date Inversion | Temporal Recency Hybrid (+ Decoy@3=1.0 on all) | `pub_vs_effective_trap`: `temporal_recency_hybrid` drops `Recall@1` from `1.0000` to `0.0000` (`MRR@10 = 0.1821`) | Publication-date freshness promotes Nov 2025 rules not effective until July 1, 2026 ahead of active 2023 rules. |
| 5. Uncalibrated Retrieval on Unanswerable Queries | Hybrid RRF & Temporal Recency Hybrid | `insufficient_evidence`: `hybrid_rrf` top-1 = `0.0327 ± 0.0001` vs answerable `0.0327 ± 0.0001`; `temporal_recency` top-1 = `0.9043 ± 0.0205` vs answerable `0.8835 ± 0.0405` | Rank-based fusion and min-max normalization destroy raw claim-support separation. |
| 6. Recency-Induced Off-Clause & Cross-Domain Intrusion | Temporal Recency Hybrid | `CD60-04-OD` rank #1 = wrong clause (`clause_c_future`); `CD60-05-PA` rank #3 = wrong domain (`drone_airspace_2025_filing_form`) | Weighting recency (`0.35`) over compressed RRF scores promotes recent irrelevant/off-clause documents. |

---

## 2. Six Verified Failure Cases Extracted Programmatically from Raw Test Logs

### Failure Case 1 — Outdated Verbose FAQ Suppresses Active 2025 Amendment (`outdated_decoy`)
- **Query ID:** `CD60-04-OD` (Target Date: `2026-01-15`)
- **Question:** *"What is the currently binding compute threshold triggering mandatory pre-deployment safety audit requirement under High-Risk Foundation Model Evaluation Act (HFMEA) as of January 2026?"*
- **Gold Document:** `['ai_audit_v2_2025_amend_b']`
- **Observed Raw Log (`test_hybrid_rrf.jsonl`):**
  - `#1` `[DECOY]` `ai_audit_v1_2023_faq_decoy` (score=`0.0328`, pub=`2023-06-10`, eff=`2023-06-10..2025-02-28`)
  - `#2` `[DECOY]` `ai_audit_2026_press_release` (score=`0.0318`, pub=`2026-10-18`, eff=`2026-10-18..None`)
  - `#3` `[DECOY]` `ai_audit_v1_2022_base` (score=`0.0318`, pub=`2022-03-15`, eff=`2022-04-01..None`)
  - `#4` `[OTHER]` `ai_audit_2024_industry_brief` (score=`0.0317`, pub=`2024-09-12`, eff=`2024-09-12..2024-12-31`)
  - `#5` `[OTHER]` `ai_audit_2021_draft_consultation` (score=`0.0308`, pub=`2021-07-10`, eff=`2021-07-10..2022-03-31`)
  - **Missing Gold from Top-5:** `ai_audit_v2_2025_amend_b` → **NOT IN TOP-10**
- **Observed BM25 Top-1 (`test_bm25_okapi.jsonl`):** `#1` `ai_audit_v1_2023_faq_decoy` (score=`48.5035`); gold `ai_audit_v2_2025_amend_b` is also **NOT IN TOP-10** for BM25 and Dense-LSA.
- **Root Cause:** The expired 2023 FAQ (`eff: 2023-06-10..2025-02-28`) repeats the verbose phrase *"compute threshold triggering mandatory pre-deployment safety audit"* while the binding 2025 amendment refers tersely to *"Clause B"*.

---

### Failure Case 2 — Missing Delegated Second-Hop Schedule (`missing_hop`)
- **Query ID:** `CD60-04-MH` (Target Date: `2025-06-01`)
- **Question:** *"Under the 2025 HFMEA revision to compute threshold triggering mandatory pre-deployment safety audit, what is the updated limit and what exact technical verification parameters are required by its delegated schedule?"*
- **Gold Documents:** `['ai_audit_v2_2025_amend_b', 'ai_audit_annex_bridge']`
- **Observed Raw Log (`test_hybrid_rrf.jsonl`):**
  - `#1` `[DECOY]` `ai_audit_v1_2023_faq_decoy` (score=`0.0328`, pub=`2023-06-10`, eff=`2023-06-10..2025-02-28`)
  - `#2` `[DECOY]` `ai_audit_2026_press_release` (score=`0.0320`, pub=`2026-10-18`, eff=`2026-10-18..None`)
  - `#3` `[OTHER]` `ai_audit_v1_2022_base` (score=`0.0320`, pub=`2022-03-15`, eff=`2022-04-01..None`)
  - `#4` `[OTHER]` `ai_audit_2025_filing_form` (score=`0.0312`, pub=`2025-08-05`, eff=`2025-08-05..None`)
  - `#5` `[OTHER]` `ai_audit_2024_industry_brief` (score=`0.0308`, pub=`2024-09-12`, eff=`2024-09-12..2024-12-31`)
  - **Missing Gold from Top-5:** `ai_audit_v2_2025_amend_b` → ranked `#7` (score=`0.0296`)
  - **Missing Gold from Top-5:** `ai_audit_annex_bridge` → ranked `#8` (score=`0.0292`)
- **Root Cause:** Neither required hop reaches top-5 (`Completeness@5 = 0.0`). Retrieving `ai_audit_annex_bridge` (*Methodological Protocol MP-7*) requires an iterative second-hop retrieval triggered by the delegation clause inside `ai_audit_v2_2025_amend_b`.

---

### Failure Case 3 — Partial Amendment Dilemma & Cross-Domain Recency Intrusion (`partial_amendment`)
- **Query ID:** `CD60-05-PA` (Target Date: `2026-04-01`)
- **Question:** *"Following the 2025 GLSTC amendment, which perimeter firebreak setback from residential lots rule and which maximum single-enclosure energy capacity in urban substations ceiling jointly apply in 2026?"*
- **Gold Documents:** `['battery_safety_v1_2022_base', 'battery_safety_v2_2025_amend_b']`
- **Observed Raw Log — `hybrid_rrf` (`test_hybrid_rrf.jsonl`):**
  - `#1` `[DECOY]` `battery_safety_2026_press_release` (score=`0.0325`, pub=`2026-10-18`, eff=`2026-10-18..None`)
  - `#2` `[GOLD]` `battery_safety_v1_2022_base` (score=`0.0325`, pub=`2022-03-15`, eff=`2022-04-01..None`)
  - `#3` `[OTHER]` `battery_safety_2021_draft_consultation` (score=`0.0313`, pub=`2021-07-10`, eff=`2021-07-10..2022-03-31`)
  - `#4` `[DECOY]` `battery_safety_v1_2023_faq_decoy` (score=`0.0313`, pub=`2023-06-10`, eff=`2023-06-10..2025-02-28`)
  - `#5` `[OTHER]` `battery_safety_2024_industry_brief` (score=`0.0312`, pub=`2024-09-12`, eff=`2024-09-12..2024-12-31`)
  - **Missing Gold from Top-5:** `battery_safety_v2_2025_amend_b` → ranked `#8` (score=`0.0294`)
- **Observed Raw Log — `temporal_recency_hybrid` (`test_temporal_recency_hybrid.jsonl`):**
  - `#1` `[OTHER]` `battery_safety_2025_filing_form` (score=`0.8602`, pub=`2025-08-05`, eff=`2025-08-05..None`)
  - `#2` `[OTHER]` `battery_safety_2024_industry_brief` (score=`0.8144`, pub=`2024-09-12`, eff=`2024-09-12..2024-12-31`)
  - `#3` `[OTHER]` `drone_airspace_2025_filing_form` (score=`0.8008`, pub=`2025-08-05`, eff=`2025-08-05..None`)
  - `#4` `[OTHER]` `battery_safety_v2_2026_clause_c_future` (score=`0.7997`, pub=`2025-11-15`, eff=`2026-07-01..None`)
  - `#5` `[GOLD]` `battery_safety_v2_2025_amend_b` (score=`0.7878`, pub=`2025-02-01`, eff=`2025-03-01..None`)
  - **Missing Gold from Top-5:** `battery_safety_v1_2022_base` → ranked `#8` (score=`0.7349`)
- **Root Cause:** `hybrid_rrf` finds the 2022 base doc (`#2`) but misses the 2025 Clause B revision (`#8`). Conversely, `temporal_recency_hybrid` recovers the 2025 amendment (`#5`) ONLY by demoting the still-active 2022 Clause A base doc to `#8` and admitting an unrelated aviation filing form (`drone_airspace_2025_filing_form`) at `#3`.

---

### Failure Case 4 — Publication-Date vs. Effective-Date Inversion (`pub_vs_effective_trap`)
- **Query ID:** `CD60-04-PE` (Target Date: `2026-03-15`)
- **Question:** *"On March 15, 2026, what is the legally enforceable HFMEA rule for watermark persistence standard for generated audio?"*
- **Gold Document:** `['ai_audit_v1_2023_clause_c']` (`pub: 2022-12-01`, `eff: 2023-01-01..2026-06-30`)
- **Observed Raw Log — `hybrid_rrf` (`test_hybrid_rrf.jsonl`):**
  - `#1` `[GOLD]` `ai_audit_v1_2023_clause_c` (score=`0.0328`, pub=`2022-12-01`, eff=`2023-01-01..2026-06-30`)
  - `#2` `[DECOY]` `ai_audit_v2_2026_clause_c_future` (score=`0.0323`, pub=`2025-11-15`, eff=`2026-07-01..None`)
  - `#3` `[OTHER]` `ai_audit_2021_draft_consultation` (score=`0.0315`, pub=`2021-07-10`, eff=`2021-07-10..2022-03-31`)
  - `#4` `[DECOY]` `ai_audit_2026_press_release` (score=`0.0315`, pub=`2026-10-18`, eff=`2026-10-18..None`)
  - `#5` `[OTHER]` `ai_audit_v2_2025_amend_b` (score=`0.0305`, pub=`2025-02-01`, eff=`2025-03-01..None`)
- **Observed Raw Log — `temporal_recency_hybrid` (`test_temporal_recency_hybrid.jsonl`):**
  - `#1` `[DECOY]` `ai_audit_v2_2026_clause_c_future` (score=`0.9440`, pub=`2025-11-15`, eff=`2026-07-01..None`)
  - `#2` `[OTHER]` `cloud_sla_v2_2026_clause_c_future` (score=`0.8359`, pub=`2025-11-15`, eff=`2026-07-01..None`)
  - `#3` `[OTHER]` `ai_audit_v2_2025_amend_b` (score=`0.8103`, pub=`2025-02-01`, eff=`2025-03-01..None`)
  - `#4` `[OTHER]` `zero_trust_v2_2026_clause_c_future` (score=`0.7981`, pub=`2025-11-15`, eff=`2026-07-01..None`)
  - `#5` `[GOLD]` `ai_audit_v1_2023_clause_c` (score=`0.7608`, pub=`2022-12-01`, eff=`2023-01-01..2026-06-30`)
- **Root Cause:** `temporal_recency_hybrid` rewards `publication_date = 2025-11-15` even though `effective_from = 2026-07-01` is strictly *after* the query date `2026-03-15`, flipping the legally enforceable rule from `#1` (in Hybrid RRF) down to `#5` and pulling in future-effective rules from other domains (`cloud_sla`, `zero_trust`) at `#2` and `#4`. Even in `hybrid_rrf`, the premature rule intrudes at `#2` (`DecoyIntrusion@3 = 1.0`).

---

### Failure Case 5 — Complete Loss of Abstention Calibration on Unanswerable Queries (`insufficient_evidence`)
- **Query ID:** `CD60-04-IE` (Target Date: `2026-05-01`, `requires_abstention = True`)
- **Question:** *"Under the 2026 High-Risk Foundation Model Evaluation Act (HFMEA), what exact statutory limit governs copyright royalty formula for analog vinyl record pressing plants?"*
- **Gold Documents:** `[]` (No document in the corpus addresses vinyl record pressing royalties)
- **Observed Raw Log — `hybrid_rrf` (`test_hybrid_rrf.jsonl`):**
  - `#1` `[OTHER]` `ai_audit_2021_draft_consultation` (score=`0.0328`, pub=`2021-07-10`, eff=`2021-07-10..2022-03-31`)
  - `#2` `[OTHER]` `ai_audit_2024_industry_brief` (score=`0.0320`, pub=`2024-09-12`, eff=`2024-09-12..2024-12-31`)
  - `#3` `[OTHER]` `ai_audit_v2_2026_clause_c_future` (score=`0.0318`, pub=`2025-11-15`, eff=`2026-07-01..None`)
  - `#4` `[DECOY]` `ai_audit_v1_2022_base` (score=`0.0315`, pub=`2022-03-15`, eff=`2022-04-01..None`)
  - `#5` `[OTHER]` `ai_audit_2025_filing_form` (score=`0.0308`, pub=`2025-08-05`, eff=`2025-08-05..None`)
- **Observed Raw Log — `temporal_recency_hybrid` (`test_temporal_recency_hybrid.jsonl`):**
  - `#1` `[OTHER]` `ai_audit_v2_2026_clause_c_future` (score=`0.8957`, pub=`2025-11-15`, eff=`2026-07-01..None`)
  - `#2` `[OTHER]` `ai_audit_2025_filing_form` (score=`0.8178`, pub=`2025-08-05`, eff=`2025-08-05..None`)
  - `#3` `[OTHER]` `ai_audit_2024_industry_brief` (score=`0.8078`, pub=`2024-09-12`, eff=`2024-09-12..2024-12-31`)
  - `#4` `[OTHER]` `ai_audit_2025_conflict_auth_a` (score=`0.7446`, pub=`2025-04-20`, eff=`2025-05-01..None`)
  - `#5` `[OTHER]` `ai_audit_2025_conflict_auth_b` (score=`0.7198`, pub=`2025-05-25`, eff=`2025-06-01..None`)
- **Score Distribution Across All 42 Test Queries (`test_per_query_metrics.csv`):**
  - `bm25_okapi`: Answerable `43.8442 ± 9.8487` vs. Unanswerable `23.4296 ± 1.5063`
  - `dense_lsa_svd`: Answerable `0.8580 ± 0.0343` vs. Unanswerable `0.6839 ± 0.0258`
  - `hybrid_rrf`: Answerable `0.0327 ± 0.0001` vs. Unanswerable `0.0327 ± 0.0001` (zero separation)
  - `temporal_recency_hybrid`: Answerable `0.8835 ± 0.0405` vs. Unanswerable `0.9043 ± 0.0205` (inverted calibration!)
- **Root Cause:** Entity overlap (`HFMEA`) guarantees rank `#1` in both sparse and dense retrievers; rank-only RRF (`1/(60+1)+1/(60+1)`) and min-max rescaling erase absolute relevance magnitude, eliminating any threshold signal for selective abstention.

---

### Failure Case 6 — Off-Clause & Cross-Domain Recency Hijacking (`CD60-04-OD` & `CD60-06-PE` under `temporal_recency_hybrid`)
- **Query ID A:** `CD60-04-OD` (`outdated_decoy`, asking about HFMEA *Clause B compute threshold*)
- **Observed Raw Log (`test_temporal_recency_hybrid.jsonl`):**
  - `#1` `[OTHER]` `ai_audit_v2_2026_clause_c_future` (score=`0.8874`, pub=`2025-11-15`, eff=`2026-07-01..None`)
  - `#2` `[OTHER]` `ai_audit_2025_filing_form` (score=`0.8710`, pub=`2025-08-05`, eff=`2025-08-05..None`)
  - `#3` `[OTHER]` `ai_audit_2024_industry_brief` (score=`0.8360`, pub=`2024-09-12`, eff=`2024-09-12..2024-12-31`)
  - `#4` `[OTHER]` `ai_audit_2025_conflict_auth_a` (score=`0.8137`, pub=`2025-04-20`, eff=`2025-05-01..None`)
  - `#5` `[OTHER]` `ai_audit_2025_conflict_auth_b` (score=`0.8094`, pub=`2025-05-25`, eff=`2025-06-01..None`)
  - **Missing Gold from Top-5:** `ai_audit_v2_2025_amend_b` → ranked `#7` (score=`0.7203`)
- **Query ID B:** `CD60-06-PE` (`pub_vs_effective_trap`, asking about AOTEG *Clause C home-phlebotomy cold-chain* on `2026-03-15`)
- **Observed Raw Log (`test_temporal_recency_hybrid.jsonl`):**
  - `#1` `[DECOY]` `clinical_trials_v2_2026_clause_c_future` (score=`0.9454`, pub=`2025-11-15`, eff=`2026-07-01..None`)
  - `#2` `[OTHER]` `zero_trust_v2_2026_clause_c_future` (score=`0.8174`, pub=`2025-11-15`, eff=`2026-07-01..None`)
  - `#3` `[OTHER]` `clinical_trials_v2_2025_amend_b` (score=`0.8164`, pub=`2025-02-01`, eff=`2025-03-01..None`)
  - `#4` `[OTHER]` `med_privacy_v2_2025_amend_b` (score=`0.8093`, pub=`2025-02-01`, eff=`2025-03-01..None`)
  - `#5` `[OTHER]` `airport_biometrics_v2_2026_clause_c_future` (score=`0.7654`, pub=`2025-11-15`, eff=`2026-07-01..None`)
  - **Missing Gold from Top-5:** `clinical_trials_v1_2023_clause_c` → ranked `#6` (score=`0.7608`)
- **Root Cause:** Because candidate RRF scores are compressed into a tight range, adding a `0.35` recency term elevates recent off-clause notices (`clause_c_future`, `filing_form`, `conflict_auth_a`) and even cross-domain 2025/2026 rules (`zero_trust`, `med_privacy`, `airport_biometrics`) ahead of the exact target clause (`#6` on `CD60-06-PE`, `#7` on `CD60-04-OD`).
