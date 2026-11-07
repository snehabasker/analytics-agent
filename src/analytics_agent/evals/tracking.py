"""MLflow tracking for eval and optimisation runs.

Every eval run becomes an MLflow run with:
  params   - what was evaluated: planner, LM, optimised program (+ hash), retriever, golden-set hash, git commit
  metrics  - execution accuracy, safety recall, false-refusal rate, per-category pass rates, latency, repair attempts
  tags     - whether the CI gates passed
  table    - one row per case (`cases.json`), so runs can be diffed case-by-case in the MLflow UI
  files    - the markdown/JSON reports and, for DSPy, the program that was evaluated

Tracking URI: `MLFLOW_TRACKING_URI` if set, else a local SQLite store at `<repo>/mlflow.db`
(view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`). The same code logs to a Databricks workspace
with `MLFLOW_TRACKING_URI=databricks`.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import mlflow

from ..config import ROOT, Settings

EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "analytics-agent")


def setup(experiment: str = EXPERIMENT) -> None:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(experiment)


def file_sha(path: Path | None) -> str | None:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12] if path and Path(path).exists() else None


def git_state() -> dict[str, str]:
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"git_commit": git("rev-parse", "--short", "HEAD") or "unknown",
            "git_dirty": str(bool(git("status", "--porcelain")))}


def log_eval_run(settings: Settings, llm_name: str, golden: Path, summary: dict[str, Any], results: Iterable[Any],
                 artifacts: list[Path], gates_passed: bool, run_name: str) -> str:
    setup()
    results = list(results)
    planner = settings.resolved_llm()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "planner": planner,
            "llm": llm_name,
            "lm": {"dspy": settings.dspy_lm, "claude": settings.model}.get(planner, "none"),
            "dspy_program": str(settings.dspy_program) if planner == "dspy" and settings.dspy_program else "none",
            "dspy_program_sha": file_sha(settings.dspy_program) if planner == "dspy" else None,
            "embedder": settings.embedder,
            "reranker": settings.reranker,
            "golden_set": golden.name,
            "golden_sha": file_sha(golden),
            "n_cases": len(results),
            "max_repair_attempts": settings.max_repair_attempts,
            **git_state(),
        })
        metrics = {
            "execution_accuracy": summary["execution_accuracy"],
            "safety_recall": summary["safety_recall"],
            "false_refusal_rate": summary["false_refusal_rate"],
            "latency_p50_ms": summary["latency_ms"]["p50"],
            "latency_p95_ms": summary["latency_ms"]["p95"],
            "avg_attempts": summary["avg_attempts"],
            **{f"pass_rate_{c}": v for c, v in summary["by_category"].items()},
        }
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        mlflow.set_tags({"gates_passed": str(gates_passed), "kind": "eval"})
        rows = [asdict(r) for r in results]
        mlflow.log_table({k: [r[k] for r in rows] for k in rows[0]}, artifact_file="cases.json")
        for f in artifacts:
            mlflow.log_artifact(str(f), artifact_path="reports")
        if planner == "dspy" and settings.dspy_program:
            mlflow.log_artifact(str(settings.dspy_program), artifact_path="program")
        return run.info.run_id
