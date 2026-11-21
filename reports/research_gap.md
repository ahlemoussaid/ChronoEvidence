# ChronoEvidence — Research Gap Analysis (Phase 1)

**Date:** 2026-11-20

## 1. Structural Matrix of Prior Work vs. ChronoEvidence (ACEC)

| Capability | HoH / FRESCO / Re3 | TimelyRAG | TEMPO | GroundRAG | Adaptive-RAG / DRAGIN | **ChronoEvidence (ACEC)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Handles outdated / conflicting documents | Yes | Yes | Partial (cross-period) | No (static) | No (static) | **Yes** |
| Preserves clause-level / partial amendment versions | No | Partial | No | No | No | **Yes (via Evidence Graph)** |
| Multi-hop claim-level coverage tracking | No | No | Benchmark only | Yes (static tri-graph) | Partial (query/token) | **Yes (`C(q)`)** |
| Explicit temporal validity & consistency check | Recency / timestamp | Hybrid temporal score | Metrics only | No | No | **Yes (`T(q)`)** |
| Budget-constrained adaptive retrieval & stopping | No | No | No | No | Yes (non-temporal) | **Yes (`C(q), T(q), U(q), B`)** |
| Claim-level citation verification & abstention | No | No | No | Chain verification | No | **Yes** |

## 2. Five Specific Research Gaps Targeted by ChronoEvidence

### Gap 1: Multi-Hop Claim Coverage Under Temporal Evolution
LC-EBG (2026) shows that multi-hop coverage—not top-1 semantic recall—is the primary failure mode of RAG, while TEMPO (2026) shows retrieval systems fail to cover all required time periods in cross-period queries. Current temporal retrievers (Re3, FRESCO, TimelyRAG) optimize document-level ranking for a single query representation rather than tracking whether *each atomic sub-claim* has temporally valid support.

### Gap 2: Recency Bias vs. Clause-Level Temporal Provenance
Re3 and FRESCO primarily target *latest-fact* conflicts where newer documents supersede older ones. However, regulatory, policy, and cross-period analytical queries (as in TimelyQABench and TEMPO) frequently require retrieving **both** an earlier baseline clause and a later partial amendment, without erroneously assuming the newest document invalidates every prior provision.

### Gap 3: Absence of Temporal Signals in Adaptive Retrieval Controllers
Existing adaptive RAG controllers decide whether to retrieve using upfront question complexity (Adaptive-RAG) or token-level generation entropy (DRAGIN). Neither mechanism detects when a model is *confidently* relying on a semantically similar but temporally obsolete passage, nor do they allocate a finite retrieval budget `B` toward specifically unsupported or temporally ambiguous claims.

### Gap 4: Unverified Grounding Under Conflicting Evidence
GroundRAG (2026) verifies intermediate reasoning chains on static benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue), and ALCE (2023) scores static citation entailment. Neither tests whether citation verification and selective abstention remain reliable when two retrieved passages are both topically relevant but hold contradictory validity intervals.

### Gap 5: Fair Budget-Controlled Evaluation
Per Laitenberger et al. (2025), iterative retrieval methods often appear stronger solely because they consume more retrieval calls and context tokens than single-pass baselines. There is a clear need for a controlled empirical study testing whether adaptive claim-level stopping yields strictly better *quality-per-retrieval-call* and *quality-per-context-token* trade-offs on temporal multi-hop tasks.

## 3. Falsifiable Research Hypothesis

> **H1 (Primary):** Under a matched retrieval budget `B` and matched context/generation budget, an adaptive retrieval controller (ACEC) that explicitly tracks weighted claim-level evidence coverage `C(q)`, temporal consistency `T(q)`, and unresolved uncertainty `U(q)` over a versioned evidence graph achieves higher multi-hop evidence completeness (`Temporal Coverage@k`, `Recall@k`, `Citation Recall`) and higher temporal validity (`Temporal Precision@k`, `Citation Precision`) than single-pass BM25, Dense, Hybrid RRF, and recency-weighted temporal RAG baselines.
>
> **H0 (Null / Falsification Condition):** Claim decomposition errors, verifier false positives, or extra retrieval overhead negate the gains of ACEC, resulting in equal or lower joint answer+evidence quality than strong Hybrid RRF or fixed-depth temporal baselines at matched cost.
