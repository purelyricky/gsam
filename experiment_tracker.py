"""
Experiment tracking for GSAM / ACE runs.

Every run always writes a local ``tracking.jsonl`` (params + flattened numeric metrics).
Set ``GSAM_TRACKER=wandb`` or ``GSAM_TRACKER=mlflow`` to also mirror to that service;
both are optional dependencies and are imported lazily.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from log_config import get_logger

logger = get_logger("tracker")


def flatten_metrics(data: Any, prefix: str = "") -> Dict[str, float]:
    """Flatten nested dicts into ``a.b.c -> number`` keeping only numeric leaves."""
    out: Dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out.update(flatten_metrics(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        out[prefix] = float(data)
    return out


class ExperimentTracker:
    """Writes run params/metrics to JSONL and, optionally, to wandb or mlflow."""

    def __init__(self, save_path: str, run_name: str, backend: Optional[str] = None):
        self.save_path = save_path
        self.run_name = run_name
        self.backend = (backend or os.environ.get("GSAM_TRACKER", "")).lower()
        self.jsonl_path = os.path.join(save_path, "tracking.jsonl")
        self._remote = None
        if self.backend in ("wandb", "mlflow"):
            self._remote = self._start_remote()

    def _start_remote(self):
        try:
            if self.backend == "wandb":
                import wandb
                return wandb.init(project=os.environ.get("WANDB_PROJECT", "gsam"), name=self.run_name)
            import mlflow
            mlflow.start_run(run_name=self.run_name)
            return mlflow
        except Exception as e:  # optional dependency or network problems must never break a run
            logger.warning(f"Tracker backend '{self.backend}' unavailable ({e}); using local JSONL only")
            return None

    def _write(self, record: Dict[str, Any]) -> None:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), "run": self.run_name, **record}
        os.makedirs(self.save_path, exist_ok=True)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_params(self, params: Dict[str, Any]) -> None:
        self._write({"type": "params", "params": params})
        if self._remote is not None:
            if self.backend == "wandb":
                self._remote.config.update(params, allow_val_change=True)
            else:
                self._remote.log_params({k: str(v)[:250] for k, v in params.items()})

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> Dict[str, float]:
        flat = flatten_metrics(metrics)
        self._write({"type": "metrics", "step": step, "metrics": flat})
        if self._remote is not None and flat:
            if self.backend == "wandb":
                self._remote.log(flat, step=step)
            else:
                self._remote.log_metrics({k.replace(" ", "_"): v for k, v in flat.items()}, step=step)
        return flat

    def finish(self) -> None:
        if self._remote is None:
            return
        try:
            if self.backend == "wandb":
                self._remote.finish()
            else:
                self._remote.end_run()
        except Exception as e:
            logger.warning(f"Failed to close tracker backend: {e}")
