#!/usr/bin/env python3
"""
Config-driven Phase 3 baseline runner for ChronoEvidence.
Executes BM25, Dense Retrieval, Hybrid RRF, and Temporal Recency Hybrid under matched budgets,
saves raw ranked outputs (JSONL + TREC), computes metrics, and appends structured logs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Ensure local src/ is importable without pip install -e .
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronoevidence.evaluation.metrics import (
    aggregate_metrics,
    cross_check_with_ir_measures,
    evaluate_query,
)
from chronoevidence.retrieval.bm25 import BM25OkapiRetriever
from chronoevidence.retrieval.dense import DenseLSARetriever, SentenceTransformerFAISSRetriever
from chronoevidence.retrieval.hybrid import HybridRRFRetriever
from chronoevidence.schemas import DocumentRecord, QueryRecord, RetrievedItem
from chronoevidence.temporal.recency import TemporalRecencyRetriever
from chronoevidence.utils import append_experiment_log, get_hardware_metadata, set_seed, sha256_file


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_retrievers(cfg: dict):
    bm25 = BM25OkapiRetriever(**cfg["bm25"])

    dense_cfg = cfg["dense"]
    backend = dense_cfg.get("backend", "dense_lsa_svd")
    if backend == "dense_lsa_svd":
        dense = DenseLSARetriever(
            embedding_dim=dense_cfg.get("embedding_dim", 96),
            seed=dense_cfg.get("seed", cfg["seed"]),
        )
    elif backend == "sentence_transformer_faiss":
        dense = SentenceTransformerFAISSRetriever(
            model_name=dense_cfg["model_name"],
            revision=dense_cfg.get("revision"),
            batch_size=dense_cfg.get("batch_size", 32),
        )
    else:
        raise ValueError(f"Unsupported dense backend: {backend}")

    # Separate instances inside Hybrid so indexing/state remain clean and inspectable
    bm25_for_hybrid = BM25OkapiRetriever(**cfg["bm25"])
    if backend == "dense_lsa_svd":
        dense_for_hybrid = DenseLSARetriever(
            embedding_dim=dense_cfg.get("embedding_dim", 96),
            seed=dense_cfg.get("seed", cfg["seed"]),
        )
    else:
        dense_for_hybrid = dense

    hybrid = HybridRRFRetriever(
        retrievers=[bm25_for_hybrid, dense_for_hybrid],
        weights=cfg["hybrid_rrf"].get("weights", [1.0, 1.0]),
        rrf_k=cfg["hybrid_rrf"].get("rrf_k", 60),
        candidate_depth=cfg["hybrid_rrf"].get("candidate_depth", 50),
    )

    temporal = TemporalRecencyRetriever(
        base_retriever=hybrid,
        alpha=cfg["temporal_recency"].get("alpha", 0.65),
        decay_lambda=cfg["temporal_recency"].get("decay_lambda", 0.35),
        reference_date=cfg["temporal_recency"].get("reference_date", "2026-11-20"),
        candidate_depth=cfg["temporal_recency"].get("candidate_depth", 40),
    )

    return [bm25, dense, hybrid, temporal]


def save_raw_run(
    out_jsonl: Path,
    out_trec: Path,
    split: str,
    method: str,
    queries: List[QueryRecord],
    runs: Dict[str, List[RetrievedItem]],
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fj, open(out_trec, "w", encoding="utf-8") as ft:
        for q in queries:
            items = runs[q.query_id]
            fj.write(
                json.dumps(
                    {
                        "query_id": q.query_id,
                        "split": split,
                        "category": q.category,
                        "question": q.question,
                        "target_date": q.target_date,
                        "gold_doc_ids": q.gold_doc_ids,
                        "outdated_decoy_ids": q.outdated_decoy_ids,
                        "requires_abstention": q.requires_abstention,
                        "method": method,
                        "retrieved": [it.model_dump() for it in items],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for it in items:
                ft.write(f"{q.query_id} Q0 {it.doc_id} {it.rank} {it.score:.6f} {method}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 retrieval baselines.")
    parser.add_argument("--config", required=True, help="Path to JSON experiment config")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    ds_dir = REPO_ROOT / cfg["dataset_dir"]
    corpus_path = ds_dir / "corpus.jsonl"
    manifest_path = ds_dir / "manifest.json"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    # Integrity check against manifest if present
    corpus_sha = sha256_file(corpus_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest.get("sha256", {}).get("corpus.jsonl")
        if expected_sha and expected_sha != corpus_sha:
            raise RuntimeError(f"Corpus SHA-256 mismatch! expected={expected_sha}, actual={corpus_sha}")

    docs = [DocumentRecord(**r) for r in load_jsonl(corpus_path)]
    doc_map = {d.doc_id: d for d in docs}

    retrievers = build_retrievers(cfg)
    index_times: Dict[str, float] = {}
    indexed_ids = set()
    for ret in retrievers:
        t0 = time.perf_counter()
        if isinstance(ret, TemporalRecencyRetriever) and id(ret.base_retriever) in indexed_ids:
            ret.index(docs, index_base=False)
        else:
            ret.index(docs)
        indexed_ids.add(id(ret))
        index_times[ret.name] = round((time.perf_counter() - t0) * 1000.0, 2)

    splits = cfg["splits"] if args.split == "all" else [args.split]
    hw_meta = get_hardware_metadata()

    for split in splits:
        q_path = ds_dir / f"queries_{split}.jsonl"
        queries = [QueryRecord(**r) for r in load_jsonl(q_path)]
        top_k = int(cfg["top_k"])

        summary_rows = []
        per_query_all = []
        category_rows = []

        for ret in retrievers:
            runs: Dict[str, List[RetrievedItem]] = {}
            latencies_ms: List[float] = []
            per_q_metrics: List[Dict[str, float]] = []

            for q in queries:
                t0 = time.perf_counter()
                items = ret.retrieve(q.question, top_k=top_k, target_date=q.target_date)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(dt_ms)
                runs[q.query_id] = items

                m = evaluate_query(q, items, doc_map)
                per_q_metrics.append(m)
                per_query_all.append(
                    {
                        "split": split,
                        "method": ret.name,
                        "query_id": q.query_id,
                        "category": q.category,
                        "domain": q.domain,
                        "requires_abstention": q.requires_abstention,
                        "latency_ms": round(dt_ms, 3),
                        "top1_doc": items[0].doc_id if items else "",
                        "top1_score": round(items[0].score, 6) if items else 0.0,
                        **m,
                    }
                )

            raw_jsonl = REPO_ROOT / "results" / "raw" / f"{split}_{ret.name}.jsonl"
            raw_trec = REPO_ROOT / "results" / "raw" / f"{split}_{ret.name}.trec"
            save_raw_run(raw_jsonl, raw_trec, split, ret.name, queries, runs)

            agg = aggregate_metrics(per_q_metrics)
            ir_check = cross_check_with_ir_measures(queries, runs)

            summary_rows.append(
                {
                    "split": split,
                    "method": ret.name,
                    "num_queries_total": len(queries),
                    "num_queries_answerable": sum(1 for q in queries if q.gold_doc_ids),
                    "index_ms": index_times[ret.name],
                    "mean_query_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 3),
                    "ir_measures_verified": ir_check is not None,
                    **{k: round(v, 4) if pd.notna(v) else None for k, v in agg.items()},
                }
            )

            # Per-category breakdown
            df_q = pd.DataFrame(
                [row for row in per_query_all if row["split"] == split and row["method"] == ret.name]
            )
            for cat, grp in df_q.groupby("category"):
                cat_agg = aggregate_metrics(
                    grp[
                        [
                            "recall@1", "recall@3", "recall@5", "recall@10",
                            "mrr@10", "ndcg@5", "ndcg@10", "hit@1", "hit@5",
                            "completeness@5", "temporal_precision@5",
                            "temporal_coverage@5", "decoy_intrusion@3",
                        ]
                    ].to_dict(orient="records")
                )
                category_rows.append(
                    {
                        "split": split,
                        "method": ret.name,
                        "category": cat,
                        "n_queries": len(grp),
                        **{k: round(v, 4) if pd.notna(v) else None for k, v in cat_agg.items()},
                    }
                )

            append_experiment_log(
                REPO_ROOT / "logs" / "experiment_log.jsonl",
                {
                    "milestone": "Phase 3 - Baseline Execution",
                    "experiment_name": cfg["experiment_name"],
                    "config_path": str(cfg_path),
                    "dataset_dir": str(cfg["dataset_dir"]),
                    "corpus_sha256": corpus_sha,
                    "split": split,
                    "seed": cfg["seed"],
                    "top_k": top_k,
                    "method": ret.name,
                    "retriever_metadata": ret.get_metadata(),
                    "hardware": hw_meta,
                    "metrics": {k: round(v, 4) if pd.notna(v) else None for k, v in agg.items()},
                    "ir_measures_cross_check": ir_check,
                    "raw_outputs": [str(raw_jsonl.relative_to(REPO_ROOT)), str(raw_trec.relative_to(REPO_ROOT))],
                },
            )

        tables_dir = REPO_ROOT / "results" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(tables_dir / f"{split}_summary_metrics.csv", index=False)
        pd.DataFrame(per_query_all).to_csv(tables_dir / f"{split}_per_query_metrics.csv", index=False)
        pd.DataFrame(category_rows).to_csv(tables_dir / f"{split}_by_category.csv", index=False)

        print(f"\n=== SUMMARY METRICS ({split.upper()} SPLIT) ===")
        print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
