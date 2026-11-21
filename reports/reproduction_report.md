# ChronoEvidence — Phase 3 Baseline Reproducibility Report

**Date:** 2026-11-20  
**Experiment ID:** `phase3_baselines_chronodiag60`  
**Config File:** `configs/baselines_chronodiag60.json`  
**Random Seed:** `42`  
**Top-k Evaluation Budget:** `k = 10` (fixed across all baselines)

---

## 1. Execution Environment & Dependency Audit (Observed Directly)

### A. Air-Gapped Sandbox Constraints
During execution, dependency installation via `pip install rank-bm25` was attempted directly and **failed due to sandbox network isolation**:
```text
CMD: /usr/local/bin/python3 -m pip install --no-input --timeout 5 --retries 1 rank-bm25
EXIT CODE: 1
STDERR: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution
```
Consequently:
1. External packages (`pyserini`, `rank-bm25`, `faiss-cpu`, `sentence-transformers`, `transformers`, `datasets`, `ir-measures`, `PyYAML`, `pytest`) could **not** be installed inside this air-gapped environment.
2. External public benchmarks (`tempo26/Tempo`, `kaist-dmlab/TimelyRAG`, `0russwest0/HoH`) could **not** be downloaded inside this sandbox.
3. Full online installation (`requirements-full.txt`), dataset ingestion (`scripts/prepare_tempo.py`), and production backend adapters (`PyseriniBM25Retriever`, `SentenceTransformerFAISSRetriever`, `cross_check_with_ir_measures`) are implemented and ready for execution on an internet/GPU-enabled machine.

### B. Executed Offline Environment
- **Python:** `3.10.16`
- **OS / Hardware:** `Linux-4.19.0-gvisor-x86_64`, 8 vCPU, 2.0 GB RAM, CPU-only (`torch 2.2.2+cu121`, `cuda_available = False`)
- **Pinned Active Libraries:** `numpy 1.26.3`, `scipy 1.12.0`, `pandas 2.0.0`, `scikit-learn 1.4.0`, `matplotlib 3.6.0`, `networkx 3.4.2`, `pydantic 2.13.4`, `torch 2.2.2`

---

## 2. Dataset Split & Integrity Verification

Because online benchmark downloads are blocked in this sandbox, Phase 3 baselines were executed on the permitted, deterministic internal diagnostic benchmark **`ChronoDiag-60` (v1.0.0)** built via `scripts/build_chronodiag60.py`.

> **Scientific Scope Disclaimer:** `ChronoDiag-60` is a controlled diagnostic stress-test suite designed to isolate six temporal and multi-hop failure regimes. Results on `ChronoDiag-60` validate pipeline correctness and failure modes, and are **NOT** substitutes for public benchmark results on `TEMPO` or `TimelyQABench`.

- **Corpus:** 150 versioned multi-domain documents across 5 domains (`healthcare`, `aviation`, `cloud_security`, `ai_governance`, `energy_battery`) — 120 versioned family documents (10 families × 12 versions/clauses/decoys) + 30 sectoral background distractors.
- **Strict Split Disjointness:** Verified by automated unit test (`test_chronodiag60_split_integrity`):
  - **Dev split:** 18 queries (3 topic families × 6 categories; 15 answerable, 3 unanswerable).
  - **Test split:** 42 queries (7 held-out topic families × 6 categories; 35 answerable, 7 unanswerable).
  - Zero overlap in `query_id` or `gold_doc_ids` between Dev and Test.
- **SHA-256 Checksums:**
  - `corpus.jsonl`: `450872bbd5e69c0c4cfab3902523bb7242558f3220e1984f17d0387cb0128cb0`
  - `queries_dev.jsonl`: `7a255f056bd495aec39becbe2fee0738d1ae96f9e0cd9a6aed6b3639b97c2775`
  - `queries_test.jsonl`: `67c0ffaa54abe542445f6a87304ccccbd275b1f2e38eb53a122935d7035cf17b`

---

## 3. Exact Commands Executed & Bugs Debugged

```bash
cd chrono-evidence
python3 scripts/build_chronodiag60.py
python3 -m unittest discover -s tests -v
python3 experiments/run_baselines.py --config configs/baselines_chronodiag60.json --split all
```

### Debugging Log (Observed & Fixed)
- **Bug Caught on First Run:** In the initial run of `experiments/run_baselines.py`, `TemporalRecencyRetriever` wrapped the already-indexed `HybridRRFRetriever`, causing `ret.index(docs)` to re-index `hybrid` a second time (`index_ms = 2248.50 ms` reported for the lightweight temporal wrapper).
- **Fix Applied:** Added `index_base: bool = True` to `TemporalRecencyRetriever.index()` and tracked indexed retriever object IDs in `run_baselines.py`. After re-running all 7 unit tests (`OK` in `0.062s`) and re-running the experiment, `temporal_recency_hybrid` indexing overhead dropped to its true metadata-mapping cost (`0.17 ms`) with identical deterministic rankings.

---

## 4. Observed Baseline Results (`ChronoDiag-60`)

All metrics below are computed directly from saved raw runs (`results/raw/*.jsonl`). Note: `Recall@k`, `MRR@10`, `nDCG@k`, `Hit@k`, `Completeness@5`, and `TemporalCoverage@5` average over answerable queries (Dev `N=15`, Test `N=35`); `TemporalPrecision@5` and `DecoyIntrusion@3` average over all queries (Dev `N=18`, Test `N=42`).

### A. Held-Out Test Split (`N = 42`; `35` Answerable)

| Method | Index (ms) | Mean Query (ms) | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | Hit@1 | Hit@5 | Completeness@5 | TempPrec@5 | TempCov@5 | Decoy@3 (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bm25_okapi` | 10.51 | 1.274 | 0.3000 | 0.5000 | 0.8000 | 0.5295 | 0.4774 | 0.5922 | 0.4000 | 0.6000 | 0.4000 | 0.3857 | 0.7000 | 0.7857 |
| `dense_lsa_svd` (96D) | 2181.65 | 8.823 | **0.3714** | 0.5000 | 0.7857 | **0.6035** | **0.5097** | **0.6203** | **0.5429** | 0.6000 | 0.4000 | 0.3476 | 0.7000 | 0.7381 |
| `hybrid_rrf` (k=60) | 2259.89 | 11.449 | 0.3000 | 0.5000 | 0.7857 | 0.5314 | 0.4774 | 0.5883 | 0.4000 | 0.6000 | 0.4000 | 0.3810 | 0.7000 | 0.7381 |
| `temporal_recency_hybrid` | 0.17* | 12.834 | 0.1286 | **0.5143** | **0.8571** | 0.3852 | 0.3610 | 0.4803 | 0.2571 | **0.6286** | 0.4000 | **0.5810** | **0.7857** | **0.3333** |

*\*Reuses indexed `hybrid_rrf` base retriever.*

### B. Development Split (`N = 18`; `15` Answerable)

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | Completeness@5 | TempPrec@5 | TempCov@5 | Decoy@3 (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bm25_okapi` | 0.2333 | 0.5000 | 0.8000 | 0.4935 | 0.4528 | 0.5675 | 0.4000 | 0.3778 | 0.7000 | 0.8333 |
| `dense_lsa_svd` (96D) | 0.3667 | 0.5333 | 0.7667 | 0.6090 | 0.5251 | 0.6111 | 0.4000 | 0.3333 | 0.7000 | 0.8333 |
| `hybrid_rrf` (k=60) | 0.3000 | 0.5333 | 0.7333 | 0.5300 | 0.4932 | 0.5708 | 0.4000 | 0.3556 | 0.7000 | 0.8333 |
| `temporal_recency_hybrid` | 0.1333 | 0.5333 | 0.8000 | 0.3879 | 0.3673 | 0.4635 | 0.3333 | 0.5222 | 0.8333 | 0.3333 |

### C. Key Empirical Takeaways
1. **No baseline exceeds `0.4000` Multi-Hop Completeness@5 on Test**, confirming that single-pass lexical, dense, hybrid, and recency-weighted pipelines leave 60% of answerable queries with incomplete evidence sets.
2. **Temporal Recency Boosting is a Double-Edged Sword:** While `temporal_recency_hybrid` reduces `DecoyIntrusion@3` from `0.7381` to `0.3333` and raises `TemporalPrecision@5` from `0.3810` to `0.5810`, it degrades `MRR@10` from `0.5314` to `0.3852` and `nDCG@10` from `0.5883` to `0.4803` because uniform publication-date recency penalizes active historical clauses (`partial_amendment`) and promotes not-yet-effective rules (`pub_vs_effective_trap`) and empty recent summaries.
3. **Rank Fusion Destroys Abstention Calibration:** On Test, `bm25_okapi` (`43.84` vs `23.43`) and `dense_lsa_svd` (`0.8580` vs `0.6839`) show lower top-1 scores on unanswerable queries than answerable ones, whereas `hybrid_rrf` assigns identical top-1 scores (`0.0327 ± 0.0001` vs `0.0327 ± 0.0001`), and `temporal_recency_hybrid` actually scores unanswerable queries *higher* (`0.9043` vs `0.8835`).

---

## 5. Saved Artifacts & Online Reproduction Guide

- **Raw Runs:** `results/raw/{dev,test}_{bm25_okapi,dense_lsa_svd,hybrid_rrf,temporal_recency_hybrid}.{jsonl,trec}`
- **Tables:** `results/tables/{dev,test}_{summary_metrics,per_query_metrics,by_category}.csv`
- **Figures:** `results/figures/baseline_comparison_test.png`, `results/figures/category_failure_heatmap_test.png`
- **Experiment Log:** `logs/experiment_log.jsonl`

### Running Full Neural Baselines on TEMPO (Internet / GPU Machine)
```bash
pip install -r requirements-full.txt
python3 scripts/prepare_tempo.py --dataset-id tempo26/Tempo --out-dir data/processed/tempo
python3 experiments/run_baselines.py --config configs/baselines_tempo_template.json --split all
```
