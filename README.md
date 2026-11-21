# ChronoEvidence

**Adaptive Evidence Retrieval for Multi-Hop Reasoning over Evolving and Conflicting Knowledge**

ChronoEvidence is an Information Retrieval and NLP research project investigating whether **ACEC (Adaptive Claim-Level Evidence Coverage)**—a query-adaptive retrieval controller tracking weighted claim coverage `C(q)`, effective-interval temporal consistency `T(q)`, and unresolved uncertainty `U(q)` under a fixed retrieval budget `B`—improves multi-hop answer grounding over strong sparse, dense, hybrid, and temporal retrieval baselines.

## Current Project Status
- **Phase 1 (Literature Verification & Benchmark Strategy):** Completed (`reports/literature_review.md`, `reports/literature_matrix.csv`, `reports/research_gap.md`, `reports/benchmark_selection.md`).
- **Phase 2 (Modular Typed Repository):** Completed (`src/chronoevidence/`, `configs/`, `tests/`, `scripts/`).
- **Phase 3 (Baselines, Diagnostics & Failure Analysis):** Completed offline on `ChronoDiag-60` (Dev/Test) with 7/7 passing unit tests, raw TREC/JSONL runs, figures, and 6 log-verified failure cases (`reports/reproduction_report.md`, `reports/failure_analysis.md`). Full online runs on `TEMPO` (`tempo26/Tempo`) + `BAAI/bge-m3` + `FAISS` are scripted and ready for GPU/internet execution.
- **Phase 4 (ACEC Controller):** Completed (`src/chronoevidence/{evidence,controller,generation}/`, 14/14 passing unit tests in `tests/test_acec.py`).
- **Phase 5 (Controlled Evaluation, Ablations & Regression Analysis):** Completed offline on `ChronoDiag-60` (`experiments/run_acec_eval.py`, `reports/experimental_results.md`, `reports/ablation_study.md`, `reports/research_contribution.md`, 3 figures, paired bootstrap CIs & Wilcoxon tests).

## Quickstart (Offline Diagnostic Reproduction)
```bash
python3 scripts/build_chronodiag60.py
python3 -m unittest discover -s tests -v
python3 experiments/run_baselines.py --config configs/baselines_chronodiag60.json --split all
```

## Full Public Benchmark Execution (Internet / GPU Machine)
```bash
pip install -r requirements-full.txt
python3 scripts/prepare_tempo.py --dataset-id tempo26/Tempo --out-dir data/processed/tempo
python3 experiments/run_baselines.py --config configs/baselines_tempo_template.json --split all
```

## Key Phase 3 Findings (`ChronoDiag-60` Test Split, `N=42`, `35` Answerable)
| Method | Recall@5 | MRR@10 | nDCG@10 | Completeness@5 | TempPrec@5 | Decoy@3 (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bm25_okapi` | 0.5000 | 0.5295 | 0.5922 | 0.4000 | 0.3857 | 0.7857 |
| `dense_lsa_svd` (96D) | 0.5000 | 0.6035 | 0.6203 | 0.4000 | 0.3476 | 0.7381 |
| `hybrid_rrf` (k=60) | 0.5000 | 0.5314 | 0.5883 | 0.4000 | 0.3810 | 0.7381 |
| `temporal_recency_hybrid` | 0.5143 | 0.3852 | 0.4803 | 0.4000 | 0.5810 | 0.3333 |

## Repository Structure
```text
chrono-evidence/
├── configs/                 # JSON experiment configs (offline ChronoDiag-60 & online TEMPO)
├── data/processed/          # Versioned splits + SHA-256 manifests (never overwrite data/raw/)
├── experiments/             # Config-driven experiment runners
├── logs/                    # Structured JSONL experiment log
├── reports/                 # Verified literature review, gap analysis, reproducibility & failure reports
├── results/                 # Raw JSONL/TREC runs, summary CSV tables, matplotlib figures
├── scripts/                 # Env inspection, ChronoDiag-60 builder, TEMPO preparer
├── src/chronoevidence/      # Modular typed IR/temporal/evaluation library
└── tests/                   # Deterministic unit & split-integrity tests
```
