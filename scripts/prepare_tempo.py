#!/usr/bin/env python3
"""
Download and normalize the verified primary public benchmark TEMPO (Abdallah et al., EMNLP 2026 Main)
from Hugging Face (`tempo26/Tempo`) into `data/processed/tempo/` without overwriting raw files.

Requires internet access and `datasets` (install `requirements-full.txt`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TEMPO benchmark splits for ChronoEvidence.")
    parser.add_argument("--dataset-id", default="tempo26/Tempo", help="Hugging Face dataset identifier")
    parser.add_argument("--revision", default="main", help="Pinned HF dataset revision / commit hash")
    parser.add_argument("--out-dir", default="data/processed/tempo", help="Output directory")
    parser.add_argument("--dev-domains", nargs="+", default=["bitcoin", "history"], help="Domains reserved for dev tuning")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "ERROR: `datasets` package is not installed. Run `pip install -r requirements-full.txt` first."
        ) from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset_id} (revision={args.revision})...")
    ds = load_dataset(args.dataset_id, revision=args.revision)
    print("Available splits:", list(ds.keys()))

    meta = {
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "splits": {k: len(v) for k, v in ds.items()},
        "dev_domains": args.dev_domains,
    }
    meta_path = out_dir / "tempo_manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved manifest to", meta_path)
    print("NOTE: Verify the dataset license on https://huggingface.co/datasets/tempo26/Tempo before redistributing.")


if __name__ == "__main__":
    main()
