# ChronoEvidence — Benchmark Selection & Data Governance Plan (Phase 1)

**Date:** 2026-11-20

## 1. Selection Criteria
1. **Verified Public Availability:** Official proceedings record + accessible repository/dataset identifier.
2. **Alignment with ChronoEvidence Core Task:** Must test at least two of: (a) multi-hop retrieval, (b) temporal validity / cross-period reasoning, (c) evolving or conflicting document versions.
3. **Official Evaluation Scripts & Fixed Splits:** Reproducible metrics without inventing incompatible custom numbers.
4. **Compute Feasibility:** Must permit local development slices while keeping official test splits untouched until final evaluation.

---

## 2. Recommended Benchmark Portfolio

### A. Primary Benchmark — **TEMPO** (Abdallah et al., EMNLP 2026 Main)
- **Source:** `https://github.com/tempo-bench/Tempo` | Hugging Face: `tempo26/Tempo` | Eval: `https://github.com/DataScienceUIBK/tempoeval`
- **Composition:** 1,730 human-authored Stack Exchange queries across 13 domains (Blockchain, Social Sciences, Applied, STEM). Crucially includes **1,605 queries decomposed into 3,976 retrieval steps with gold document mappings** for step-wise multi-hop evaluation.
- **Official Metrics:** Standard IR (`nDCG@k`, `Recall@k`), plus `Temporal Coverage@k` (cross-period completeness) and `Temporal Precision@k`.
- **Why Primary:** It is the only verified 2026 main-conference benchmark that natively combines **multi-hop step decompositions**, **cross-period temporal reasoning**, **multi-domain generalization** (13 domains, enabling Experiment F: cross-domain tuning vs. held-out domains), and **official temporal coverage metrics**.
- **License & Governance Check Required Before Use:** Inspect `tempo26/Tempo` dataset card and `tempo-bench/Tempo` repository license upon local download (note Stack Exchange source content is typically CC BY-SA; confirm attribution and redistribution rules). Also note `temporal_metrics.py` uses an LLM judge—pin exact open-weight/API judge model and log prompt + seed, while always reporting deterministic gold-ID `Recall@k`, `MRR`, `nDCG@10`, and step-level gold coverage alongside LLM-judged metrics.

### B. Secondary Benchmark — **TimelyQABench** (TimelyRAG, Nam et al., EMNLP 2026 Findings) *with* **HoH** (Ouyang et al., ACL 2025 Main) as Conflict Fallback
- **Primary Secondary Choice: TimelyQABench** (`https://github.com/kaist-dmlab/TimelyRAG`)
  - **Why:** Specifically built for time-critical QA over **overlapping, evolving regulatory documents**—the closest match to ChronoEvidence's partial-amendment and clause-supersession setting.
  - **Verification Gate:** Confirm exact downloadability, dataset size, official splits, and license immediately upon cloning `kaist-dmlab/TimelyRAG`.
- **Verified Contingency Secondary Choice: HoH** (`https://github.com/0russwest0/HoH`, ACL 2025 Main)
  - **Why:** If TimelyQABench has restricted data access or heavy preprocessing barriers, **HoH** provides a verified, open ACL 2025 benchmark with explicit current vs. outdated Wikipedia evidence annotations and a mock search engine.
  - **Bonus Re-Ranking Diagnostic:** If FRESCO (`arXiv:2604.14227`) releases its 7,316-query Wikipedia-revision candidate pools publicly, it can serve as an optional re-ranker conflict ablation.

### C. Small Hand-Audited Diagnostic Evaluation Set — **ChronoDiag-60** (Internal Controlled Stress Test)
- **Purpose:** Public benchmarks rarely isolate all six fine-grained failure modes cleanly with ground-truth clause-level `SUPERSEDES` / `MODIFIES` / `CONTRADICTS` relations and explicit abstention labels.
- **Design:** Construct a compact, version-controlled, fully auditable **60-query diagnostic suite** (10 queries per failure mode) over synthetic/public-domain versioned policy & technical-spec documents (explicitly labeled as a *diagnostic unit/stress test*, NEVER presented as a substitute for TEMPO/TimelyQABench):
  1. **Missing Hop (10 queries):** Multi-claim question where one claim is lexically obvious and the second requires an indirect bridge document.
  2. **Outdated High-Similarity Decoy (10 queries):** Obsolete 2022–2023 passage shares high lexical/embedding overlap with query while a terse 2025–2026 amendment updates the fact.
  3. **Partial Amendment (10 queries):** Newer document modifies *Clause B* while leaving *Clause A* intact; tests that the controller preserves *Clause A* rather than discarding the entire original document.
  4. **Publication vs. Effective Date Trap (10 queries):** Document published in Nov 2025 takes effect in July 2026; query asks what applies in March 2026.
  5. **True Unresolved Conflict (10 queries):** Two authoritative documents with overlapping validity intervals state contradictory values; system must expose the conflict rather than hallucinate consensus.
  6. **Unanswerable / Insufficient Evidence (10 queries):** Required claim has no valid supporting passage in corpus; tests selective abstention and budget exhaustion stopping.

---

## 3. Split Discipline & Anti-Leakage Rules
1. Never tune ACEC thresholds (`tau_C`, `tau_T`, `tau_U`) on test splits. Use a dedicated validation subset (e.g., 2–3 TEMPO domains as dev/tuning and remaining domains as held-out test for cross-domain evaluation).
2. Store immutable checksums (`sha256`) and Hugging Face commit hashes for every dataset artifact in `data/README.md`.
3. Never overwrite raw downloaded benchmark files; write all normalized records to `data/processed/`.
