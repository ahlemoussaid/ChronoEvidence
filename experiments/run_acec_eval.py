#!/usr/bin/env python3
"""
Phase 5 Controlled Evaluation & Ablation Runner for ChronoEvidence.

Evaluates:
  1. Four Phase 3 baselines (BM25, Dense-LSA, Hybrid RRF, Temporal Recency) under matched generation budget.
  2. ACEC across retrieval budgets B in {1, 2, 3, 4} (including B=1 matched single-call budget).
  3. Five isolated ACEC ablations at B=3:
       - acec_no_decomposition
       - acec_no_coverage
       - acec_no_temporal
       - acec_no_uncertainty
       - acec_fixed_stopping
  4. Paired statistical comparisons (paired bootstrap 95% CIs, Wilcoxon signed-rank p-values,
     and explicit Win / Tie / Regression counts per metric).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronoevidence.controller.acec import ACECConfig, ACECController
from chronoevidence.evaluation.metrics import aggregate_metrics, evaluate_query
from chronoevidence.generation.grounded_extractive import GroundedExtractiveGenerator
from chronoevidence.retrieval.bm25 import BM25OkapiRetriever
from chronoevidence.retrieval.dense import DenseLSARetriever
from chronoevidence.retrieval.hybrid import HybridRRFRetriever
from chronoevidence.schemas import DocumentRecord, QueryRecord
from chronoevidence.temporal.recency import TemporalRecencyRetriever
from chronoevidence.utils import append_experiment_log, get_hardware_metadata, set_seed, sha256_file


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_grounding(
    query: QueryRecord,
    abstained: bool,
    cited_doc_ids: List[str],
    flagged_conflict: bool,
) -> Dict[str, float]:
    """Compute claim/citation grounding, abstention, and conflict-detection metrics."""
    gold_set = set(query.gold_doc_ids)
    cited_set = set(cited_doc_ids)

    if query.requires_abstention:
        abstention_correct = 1.0 if abstained else 0.0
        false_abstention = float("nan")
        unsupported_answer = 0.0 if abstained else 1.0
        cit_prec = float("nan")
        cit_rec = float("nan")
        joint_score = 1.0 if abstained else 0.0
    else:
        abstention_correct = 1.0 if not abstained else 0.0
        false_abstention = 1.0 if abstained else 0.0
        unsupported_answer = float("nan")
        if abstained or not cited_set:
            cit_prec = 0.0
            cit_rec = 0.0
            joint_score = 0.0
        else:
            tp = len(cited_set & gold_set)
            cit_prec = tp / len(cited_set)
            cit_rec = tp / len(gold_set)
            joint_score = (2 * cit_prec * cit_rec / (cit_prec + cit_rec)) if (cit_prec + cit_rec) > 0 else 0.0

    conflict_tp = 1.0 if (query.has_true_conflict and flagged_conflict) else (0.0 if query.has_true_conflict else float("nan"))
    conflict_fp = 1.0 if ((not query.has_true_conflict) and flagged_conflict) else (0.0 if not query.has_true_conflict else float("nan"))

    return {
        "citation_precision": cit_prec,
        "citation_recall": cit_rec,
        "joint_grounding_score": joint_score,
        "abstention_accuracy": abstention_correct,
        "false_abstention_rate": false_abstention,
        "unsupported_answer_rate": unsupported_answer,
        "conflict_recall": conflict_tp,
        "conflict_false_positive": conflict_fp,
    }


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> Tuple[float, float, float, float]:
    """Return (mean_diff, ci_low, ci_high, wilcoxon_p) for paired arrays a - b."""
    mask = (~np.isnan(a)) & (~np.isnan(b))
    a_v, b_v = a[mask], b[mask]
    if len(a_v) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    diff = a_v - b_v
    mean_d = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = np.mean(diff[idx], axis=1)
    lo, hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    if np.allclose(diff, 0.0):
        p_val = 1.0
    else:
        try:
            p_val = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
        except Exception:
            p_val = float("nan")
    return mean_d, lo, hi, p_val


def run_split(
    split: str,
    docs: List[DocumentRecord],
    doc_map: Dict[str, DocumentRecord],
    queries: List[QueryRecord],
    cfg: dict,
    corpus_sha: str,
    hw_meta: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_k = int(cfg["top_k"])
    gen = GroundedExtractiveGenerator(**cfg["generation_budget"])

    # Build base retrievers
    bm25 = BM25OkapiRetriever(**cfg["bm25"])
    dense = DenseLSARetriever(embedding_dim=cfg["dense"]["embedding_dim"], seed=cfg["seed"])
    hybrid = HybridRRFRetriever(
        retrievers=[
            BM25OkapiRetriever(**cfg["bm25"]),
            DenseLSARetriever(embedding_dim=cfg["dense"]["embedding_dim"], seed=cfg["seed"]),
        ],
        weights=cfg["hybrid_rrf"]["weights"],
        rrf_k=cfg["hybrid_rrf"]["rrf_k"],
        candidate_depth=cfg["hybrid_rrf"]["candidate_depth"],
    )
    temporal = TemporalRecencyRetriever(
        base_retriever=hybrid,
        alpha=cfg["temporal_recency"]["alpha"],
        decay_lambda=cfg["temporal_recency"]["decay_lambda"],
        reference_date=cfg["temporal_recency"]["reference_date"],
        candidate_depth=cfg["temporal_recency"]["candidate_depth"],
    )

    bm25.index(docs)
    dense.index(docs)
    hybrid.index(docs)
    temporal.index(docs, index_base=False)

    # Build ACEC variants sharing the already-indexed Hybrid RRF base retriever (no redundant re-indexing)
    base_acec = cfg["acec_default"]
    acec_specs: List[Tuple[str, ACECConfig]] = []
    for B in cfg["budget_sweep"]:
        label = "acec_full" if B == base_acec["budget_calls"] else f"acec_budget_B{B}"
        acec_specs.append((label, ACECConfig(**{**base_acec, "budget_calls": B})))

    # Five required ablations at default budget B=3
    acec_specs.extend(
        [
            ("ablation_no_decomposition", ACECConfig(**{**base_acec, "use_decomposition": False})),
            ("ablation_no_coverage", ACECConfig(**{**base_acec, "use_coverage": False})),
            ("ablation_no_temporal", ACECConfig(**{**base_acec, "use_temporal": False})),
            ("ablation_no_uncertainty", ACECConfig(**{**base_acec, "use_uncertainty": False})),
            ("ablation_fixed_stopping", ACECConfig(**{**base_acec, "use_adaptive_stopping": False})),
        ]
    )

    systems: List[Tuple[str, str, object]] = [
        ("bm25_okapi", "baseline", bm25),
        ("dense_lsa_svd", "baseline", dense),
        ("hybrid_rrf", "baseline", hybrid),
        ("temporal_recency_hybrid", "baseline", temporal),
    ]
    for name, a_cfg in acec_specs:
        ctrl = ACECController(base_retriever=hybrid, config=a_cfg, name=name)
        ctrl.index(docs, index_base=False)
        systems.append((name, "acec", ctrl))

    summary_rows = []
    per_query_rows = []
    category_rows = []

    raw_dir = REPO_ROOT / "results" / "raw" / "phase5"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for sys_name, sys_type, model in systems:
        raw_records = []
        per_q_metrics = []
        latencies = []
        calls_list = []
        docs_inspected_list = []
        ctx_tokens_list = []

        for q in queries:
            t0 = time.perf_counter()
            if sys_type == "baseline":
                items = model.retrieve(q.question, top_k=top_k, target_date=q.target_date)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                ans = gen.generate(items, doc_map, abstain=False, explicit_citations=None, flagged_conflict=False)
                calls_used = 1
                docs_inspected = len(items)
                c_q = float("nan")
                t_q = float("nan")
                u_q = float("nan")
                stop_reason = "single_pass"
                trace_dump = []
            else:
                res = model.run(q.question, top_k=top_k, target_date=q.target_date)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                items = res.retrieved
                flat_cites = [d for cl in res.claim_citations.values() for d in cl]
                ans = gen.generate(
                    items,
                    doc_map,
                    abstain=res.abstain,
                    explicit_citations=flat_cites if flat_cites else None,
                    flagged_conflict=res.has_conflict,
                )
                calls_used = res.retrieval_calls
                docs_inspected = res.unique_docs_inspected
                c_q, t_q, u_q = res.C_q, res.T_q, res.U_q
                stop_reason = res.stop_reason
                trace_dump = [s.model_dump() for s in res.trace]

            ret_m = evaluate_query(q, items, doc_map)
            gnd_m = evaluate_grounding(q, ans.abstained, ans.cited_doc_ids, ans.flagged_conflict)
            combined_m = {**ret_m, **gnd_m}
            per_q_metrics.append(combined_m)

            latencies.append(dt_ms)
            calls_list.append(calls_used)
            docs_inspected_list.append(docs_inspected)
            ctx_tokens_list.append(ans.context_tokens)

            per_query_rows.append(
                {
                    "split": split,
                    "method": sys_name,
                    "system_type": sys_type,
                    "query_id": q.query_id,
                    "category": q.category,
                    "domain": q.domain,
                    "requires_abstention": q.requires_abstention,
                    "has_true_conflict": q.has_true_conflict,
                    "abstained": ans.abstained,
                    "flagged_conflict": ans.flagged_conflict,
                    "retrieval_calls": calls_used,
                    "unique_docs_inspected": docs_inspected,
                    "context_tokens": ans.context_tokens,
                    "latency_ms": round(dt_ms, 3),
                    "C_q": c_q,
                    "T_q": t_q,
                    "U_q": u_q,
                    "stop_reason": stop_reason,
                    "cited_doc_ids": ",".join(ans.cited_doc_ids),
                    "top3_docs": ",".join(it.doc_id for it in items[:3]),
                    **combined_m,
                }
            )
            raw_records.append(
                {
                    "query_id": q.query_id,
                    "split": split,
                    "category": q.category,
                    "question": q.question,
                    "target_date": q.target_date,
                    "gold_doc_ids": q.gold_doc_ids,
                    "requires_abstention": q.requires_abstention,
                    "has_true_conflict": q.has_true_conflict,
                    "method": sys_name,
                    "retrieval_calls": calls_used,
                    "unique_docs_inspected": docs_inspected,
                    "context_tokens": ans.context_tokens,
                    "abstained": ans.abstained,
                    "flagged_conflict": ans.flagged_conflict,
                    "cited_doc_ids": ans.cited_doc_ids,
                    "C_q": None if math.isnan(c_q) else c_q,
                    "T_q": None if math.isnan(t_q) else t_q,
                    "U_q": None if math.isnan(u_q) else u_q,
                    "stop_reason": stop_reason,
                    "trace": trace_dump,
                    "retrieved": [it.model_dump() for it in items],
                }
            )

        out_raw = raw_dir / f"{split}_{sys_name}.jsonl"
        with open(out_raw, "w", encoding="utf-8") as f:
            for rec in raw_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        agg = aggregate_metrics(per_q_metrics)
        summary_rows.append(
            {
                "split": split,
                "method": sys_name,
                "system_type": sys_type,
                "mean_calls": round(float(np.mean(calls_list)), 2),
                "mean_docs_inspected": round(float(np.mean(docs_inspected_list)), 2),
                "mean_ctx_tokens": round(float(np.mean(ctx_tokens_list)), 1),
                "mean_query_ms": round(float(np.mean(latencies)), 3),
                **{k: round(v, 4) if pd.notna(v) else None for k, v in agg.items()},
            }
        )

        # Category breakdown
        df_sub = pd.DataFrame([r for r in per_query_rows if r["split"] == split and r["method"] == sys_name])
        metric_cols = list(per_q_metrics[0].keys())
        for cat, grp in df_sub.groupby("category"):
            c_agg = aggregate_metrics(grp[metric_cols].to_dict(orient="records"))
            category_rows.append(
                {
                    "split": split,
                    "method": sys_name,
                    "category": cat,
                    "n_queries": len(grp),
                    "mean_calls": round(float(grp["retrieval_calls"].mean()), 2),
                    "mean_ctx_tokens": round(float(grp["context_tokens"].mean()), 1),
                    "mean_query_ms": round(float(grp["latency_ms"].mean()), 3),
                    **{k: round(v, 4) if pd.notna(v) else None for k, v in c_agg.items()},
                }
            )

        append_experiment_log(
            REPO_ROOT / "logs" / "experiment_log.jsonl",
            {
                "milestone": "Phase 5 - ACEC Evaluation & Ablations",
                "experiment_name": cfg["experiment_name"],
                "split": split,
                "seed": cfg["seed"],
                "corpus_sha256": corpus_sha,
                "method": sys_name,
                "mean_calls": round(float(np.mean(calls_list)), 2),
                "mean_ctx_tokens": round(float(np.mean(ctx_tokens_list)), 1),
                "mean_query_ms": round(float(np.mean(latencies)), 3),
                "hardware": hw_meta,
                "metrics": {k: round(v, 4) if pd.notna(v) else None for k, v in agg.items()},
                "raw_output": str(out_raw.relative_to(REPO_ROOT)),
            },
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(per_query_rows), pd.DataFrame(category_rows)


def compute_paired_comparisons(df_pq: pd.DataFrame, target_method: str, comparators: List[str], seed: int = 42) -> pd.DataFrame:
    """Compute paired deltas, 95% bootstrap CIs, Wilcoxon p-values, and Win/Tie/Regression counts."""
    metrics = [
        ("recall@5", True),
        ("mrr@10", True),
        ("ndcg@10", True),
        ("completeness@5", True),
        ("temporal_precision@5", True),
        ("temporal_coverage@5", True),
        ("decoy_intrusion@3", False),  # lower is better
        ("citation_precision", True),
        ("citation_recall", True),
        ("joint_grounding_score", True),
        ("abstention_accuracy", True),
        ("latency_ms", False),  # lower is better
        ("retrieval_calls", False),  # lower is better
        ("context_tokens", False),  # lower is better
    ]
    t_df = df_pq[df_pq["method"] == target_method].sort_values("query_id").reset_index(drop=True)
    rows = []
    for comp in comparators:
        c_df = df_pq[df_pq["method"] == comp].sort_values("query_id").reset_index(drop=True)
        assert list(t_df["query_id"]) == list(c_df["query_id"])
        for m, higher_is_better in metrics:
            a = t_df[m].to_numpy(dtype=float)
            b = c_df[m].to_numpy(dtype=float)
            mask = (~np.isnan(a)) & (~np.isnan(b))
            a_v, b_v = a[mask], b[mask]
            mean_d, lo, hi, p_val = paired_bootstrap_ci(a, b, n_boot=2000, seed=seed)
            if higher_is_better:
                wins = int(np.sum(a_v > b_v + 1e-9))
                regs = int(np.sum(a_v < b_v - 1e-9))
            else:
                wins = int(np.sum(a_v < b_v - 1e-9))
                regs = int(np.sum(a_v > b_v + 1e-9))
            ties = int(len(a_v) - wins - regs)
            rows.append(
                {
                    "target": target_method,
                    "comparator": comp,
                    "metric": m,
                    "higher_is_better": higher_is_better,
                    "n_paired": int(len(a_v)),
                    "target_mean": round(float(np.mean(a_v)), 4),
                    "comparator_mean": round(float(np.mean(b_v)), 4),
                    "delta_target_minus_comp": round(mean_d, 4),
                    "ci_95_low": round(lo, 4),
                    "ci_95_high": round(hi, 4),
                    "wilcoxon_p": round(p_val, 5) if not math.isnan(p_val) else None,
                    "wins": wins,
                    "ties": ties,
                    "regressions": regs,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 ACEC evaluation & ablations.")
    parser.add_argument("--config", default="configs/acec_chronodiag60.json")
    args = parser.parse_args()

    cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    ds_dir = REPO_ROOT / cfg["dataset_dir"]
    corpus_path = ds_dir / "corpus.jsonl"
    corpus_sha = sha256_file(corpus_path)
    docs = [DocumentRecord(**r) for r in load_jsonl(corpus_path)]
    doc_map = {d.doc_id: d for d in docs}
    hw_meta = get_hardware_metadata()

    tables_dir = REPO_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for split in ["dev", "test"]:
        queries = [QueryRecord(**r) for r in load_jsonl(ds_dir / f"queries_{split}.jsonl")]
        df_sum, df_pq, df_cat = run_split(split, docs, doc_map, queries, cfg, corpus_sha, hw_meta)
        df_sum.to_csv(tables_dir / f"phase5_{split}_summary.csv", index=False)
        df_pq.to_csv(tables_dir / f"phase5_{split}_per_query.csv", index=False)
        df_cat.to_csv(tables_dir / f"phase5_{split}_by_category.csv", index=False)

        if split == "test":
            df_paired_base = compute_paired_comparisons(
                df_pq,
                target_method="acec_full",
                comparators=["bm25_okapi", "dense_lsa_svd", "hybrid_rrf", "temporal_recency_hybrid"],
                seed=int(cfg["seed"]),
            )
            df_paired_base.to_csv(tables_dir / "phase5_test_paired_vs_baselines.csv", index=False)

            df_paired_abl = compute_paired_comparisons(
                df_pq,
                target_method="acec_full",
                comparators=[
                    "acec_budget_B1",
                    "ablation_no_decomposition",
                    "ablation_no_coverage",
                    "ablation_no_temporal",
                    "ablation_no_uncertainty",
                    "ablation_fixed_stopping",
                ],
                seed=int(cfg["seed"]),
            )
            df_paired_abl.to_csv(tables_dir / "phase5_test_paired_vs_ablations.csv", index=False)

        print(f"\n=== PHASE 5 SUMMARY ({split.upper()} SPLIT) ===")
        cols = [
            "method", "mean_calls", "mean_ctx_tokens", "mean_query_ms",
            "recall@5", "mrr@10", "ndcg@10", "completeness@5",
            "temporal_precision@5", "decoy_intrusion@3",
            "citation_precision", "citation_recall", "joint_grounding_score",
            "abstention_accuracy", "false_abstention_rate", "unsupported_answer_rate"
        ]
        print(df_sum[cols].to_string(index=False))


if __name__ == "__main__":
    main()
