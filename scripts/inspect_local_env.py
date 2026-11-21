#!/usr/bin/env python3
"""Inspect local hardware, Python environment, Git state, and dataset readiness for ChronoEvidence."""

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def run_cmd(cmd: list[str]) -> dict:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def main() -> None:
    packages = [
        "torch", "transformers", "datasets", "sentence_transformers",
        "faiss", "pyserini", "rank_bm25", "ir_measures", "networkx",
        "sklearn", "scipy", "numpy", "pandas", "matplotlib", "pytest",
        "yaml", "pydantic", "nltk", "mlflow", "wandb"
    ]
    pkg_status = {}
    for pkg in packages:
        if importlib.util.find_spec(pkg) is not None:
            try:
                mod = importlib.import_module(pkg)
                pkg_status[pkg] = getattr(mod, "__version__", "installed")
            except Exception as exc:
                pkg_status[pkg] = f"import_error: {exc}"
        else:
            pkg_status[pkg] = "NOT_INSTALLED"

    total, used, free = shutil.disk_usage(".")
    cuda_info = {"available": False}
    if pkg_status.get("torch") not in ("NOT_INSTALLED", None):
        try:
            import torch
            cuda_info = {
                "available": bool(torch.cuda.is_available()),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        except Exception as exc:
            cuda_info = {"available": False, "error": str(exc)}

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "disk_gb": {
            "total": round(total / (1024**3), 2),
            "used": round(used / (1024**3), 2),
            "free": round(free / (1024**3), 2),
        },
        "cuda": cuda_info,
        "nvidia_smi": run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]) if shutil.which("nvidia-smi") else "NOT_FOUND",
        "git_version": run_cmd(["git", "--version"]) if shutil.which("git") else "NOT_FOUND",
        "git_status": run_cmd(["git", "status", "--short"]) if shutil.which("git") else "NOT_FOUND",
        "packages": pkg_status,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
