# ChronoEvidence — Research Contribution & Scientific Assessment

**Date:** 2026-11-20

## 1. What Existing Methods Cannot Adequately Address
1. **Single-pass lexical, dense, and hybrid retrievers (`BM25`, `Dense`, `Hybrid RRF`)** optimize document-level topical similarity. They fail when a query requires multiple distinct hops (`missing_hop`), when a verbose superseded document outscores a terse amendment (*Evolving Semantic Conflict*), when a newer amendment modifies only one clause while an older clause stays active (`partial_amendment`), or when entity overlap masks an unanswerable question (`insufficient_evidence`).
2. **Publication-date recency baselines (`TemporalRecencyHybrid`, cf. freshness heuristics)** conflate publication timestamp with legal/factual effective validity—demoting active historical clauses (`partial_amendment`) and promoting not-yet-effective rules (`pub_vs_effective_trap`) or recent off-clause notices.
3. **Existing adaptive RAG controllers (`Adaptive-RAG`, `DRAGIN`, `Self-RAG`)** do not track per-claim effective-interval temporal validity or clause-level provenance edges (`MODIFIES` vs. `SUPERSEDES` vs. `REFERS_TO`).

## 2. Why That Limitation Matters
In high-stakes domains (regulatory compliance, clinical protocols, financial/legal policies, evolving technical specifications), returning a semantically similar but expired clause, omitting a delegated technical annex, or confidently answering an unanswerable query produces **verifiably wrong, high-confidence hallucinations** even when the underlying LLM reads its context faithfully.

## 3. What ACEC Changes
**ACEC (Adaptive Claim-Level Evidence Coverage)** replaces single-pass relevance ranking with a closed-loop, budget-constrained retrieval controller that:
1. Decomposes compound questions into weighted atomic claims $Q = \{(c_i, w_i)\}$.
2. Maintains a NetworkX provenance graph (`SUPERSEDES`, `MODIFIES`, `CONFLICTS_WITH`, `REFERS_TO`) with explicit effective validity intervals (`effective_from`, `effective_to`).
3. Explicitly evaluates claim-level support $s_i$ (rejecting non-substantive drafts/commentaries and pure entity-overlap decoys) to compute weighted coverage $C(q) = \frac{\sum w_i s_i}{\sum w_i}$, effective-interval temporal consistency $T(q)$, and unresolved uncertainty $U(q)$.
4. Issues targeted follow-up retrieval queries within a strict call budget $B$, stops adaptively when $(C(q) \ge \tau_C, T(q) \ge \tau_T, U(q) \le \tau_U)$, and selectively abstains when support remains insufficient.

## 4. What Evidence Supports the Proposed Improvement
On the held-out `ChronoDiag-60` Test split (`N = 42`, `35` answerable, `seed = 42`) under a matched generation context budget (`max_passages = 5`, `max_context_tokens = 450`):
- **Multi-hop completeness doubled:** `Completeness@5` increased from `0.4000` (all 4 baselines) to **`0.8000`** (`+0.4000`, 95% CI `[+0.2564, +0.5714]`, Wilcoxon `p = 0.00018`).
- **Ranking & temporal validity significantly improved:** `nDCG@10` rose from `0.5883` (`hybrid_rrf`) and `0.6203` (`dense_lsa_svd`) to **`0.8000`** (`p <= 0.0021`); `Temporal Precision@5` rose from `0.3810` (`hybrid_rrf`) and `0.5810` (`temporal_recency_hybrid`) to **`0.7810`** (`p <= 0.00002`); `Temporal Coverage@5` reached **`1.0000`** (`p <= 0.0024`).
- **Unsupported answers eliminated:** `Unsupported Answer Rate` dropped from `1.0000` (all 4 baselines) to **`0.0000`**, lifting `Joint Grounding Score` from `0.3611` (`hybrid_rrf`) to **`0.6667`** (`acec_full`, `p = 0.00005`) and **`0.7778`** (`acec_budget_B2`).
- **Adaptive stopping efficiency validated:** Saved `27.7%` retrieval calls (`2.17` vs `3.00`) and `29.3%` latency (`22.97 ms` vs `32.48 ms`) vs. fixed-depth stopping with identical `nDCG@10` and `Completeness@5`.

## 5. Which Hypotheses Were Rejected, Partially Falsified, or Remain Untested
- **Partially Falsified — Single-Call Parity (`B = 1`):** Under a strictly matched 1-call retrieval budget (`acec_budget_B1`), ACEC did **not** beat `dense_lsa_svd` or `hybrid_rrf` on `nDCG@10` (`0.5226` vs `0.6203`/`0.5883`); ACEC's retrieval gains genuinely require at least `B = 2..3` adaptive calls.
- **Rejected — Current Sub-Query Reformulation on `outdated_decoy`:** Appending full verbose predicate phrases to amendment sub-queries diluted retrieval (`0` new docs added on 7/7 `outdated_decoy` Test queries, `Recall@5 = 0.0000`), showing that lexical sub-query expansion alone does not solve *Evolving Semantic Conflict* without either query-pruning or corpus-level version-link indexing.
- **Rejected — Additive Uncalibrated Uncertainty Penalties:** Combining un-scoped cross-family delegation checks (`+0.25`) and naive annex staleness (`+0.35`) caused `7/35` (`20%`) false abstentions on `missing_hop` queries where retrieval was 100% complete (`C_q = 0.9763, T_q = 1.0000`).
- **Untested (Pending Online/GPU Execution):** Performance on large-scale public benchmarks (`TEMPO` 13 domains, `TimelyQABench`, `HoH`), neural dense encoders (`BAAI/bge-m3` + `FAISS`), NLI/cross-encoder claim verifiers, and full LLM grounded generation (`Qwen3-8B`).

## 6. Promising Future Research Directions (Internship Continuation Roadmap)
1. **Calibrated & Learned Uncertainty / Stopping Controller:** Replace additive heuristic penalties in $U(q)$ with a lightweight calibrated classifier/regressor predicting marginal retrieval utility and abstention probability from $(C_t, T_t, \text{graph features}, \text{score margin})$, scoped strictly to query-relevant document families.
2. **Concise Entity-Anchored Version Discovery for Evolving Semantic Conflict:** Decouple amendment search queries from verbose superseded clause phrasing (or integrate lightweight corpus-level `SUPERSEDES`/`MODIFIES` graph indexing) to resolve the `0.0000` recall failure on `outdated_decoy`.
3. **Public Multi-Domain Scaling (`TEMPO` + `TimelyQABench`):** Evaluate ACEC with `BAAI/bge-m3`, `BGE-reranker`, and open-weight LLM claim verification across `TEMPO`'s 13 domains (training/tuning on 2–3 domains, testing zero-shot generalization on 10 held-out domains).
4. **Cross-Lingual Temporal Evidence Grounding:** Extend evaluation to English/French/Arabic temporal corpora to decouple cross-lingual retrieval loss from temporal reasoning errors.
