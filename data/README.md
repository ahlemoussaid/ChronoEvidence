# ChronoEvidence Data Governance & Version Tracking

1. **Raw datasets (`data/raw/`)** are never modified in place and are excluded from Git tracking.
2. **Permitted Internal Diagnostic Split (`data/processed/chronodiag60/`)**:
   - Generated deterministically via `python scripts/build_chronodiag60.py`.
   - 150 versioned multi-domain documents + 60 queries (Dev: 18 queries across 3 topic families; Test: 42 queries across 7 held-out topic families).
   - Explicitly labeled as a controlled diagnostic stress-test suite; NEVER substitute for public benchmarks (`TEMPO`, `TimelyQABench`, `HoH`).
3. **Public Benchmarks (Online Execution)**:
   - Primary: **TEMPO** (`tempo26/Tempo`, EMNLP 2026 Main) — prepare via `python scripts/prepare_tempo.py`.
   - Secondary: **TimelyQABench** (`github.com/kaist-dmlab/TimelyRAG`, EMNLP 2026 Findings) or fallback **HoH** (`github.com/0russwest0/HoH`, ACL 2025 Main).
