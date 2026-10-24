"""
Shared helpers for checkpoint / resume handling used by the ACE and GSAM orchestrators.
"""
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional


def find_latest_checkpoint(directory: str, extension: str) -> Optional[str]:
    """Return the file in ``directory`` with the highest trailing step number, or None."""
    best_step = -1
    best_file = None
    try:
        for fname in os.listdir(directory):
            if not fname.endswith(extension):
                continue
            nums = list(map(int, re.findall(r'\d+', fname)))
            if nums:
                step = nums[-1]
                if step > best_step:
                    best_step = step
                    best_file = os.path.join(directory, fname)
    except (FileNotFoundError, OSError):
        pass
    return best_file


def write_progress(save_path: str, timestamp: Optional[str] = None, **kwargs) -> None:
    """Write progress.json so runs can be resumed after interruption."""
    progress = {"timestamp": timestamp or datetime.now().isoformat(), **kwargs}
    with open(os.path.join(save_path, "progress.json"), "w") as f:
        json.dump(progress, f, indent=2)


def default_resume_state() -> Dict[str, Any]:
    return {
        "start_epoch": 1, "start_step": 0,
        "start_window": 0, "start_global_step": 0,
        "prior_online_results": None,
    }


def apply_progress_files(resume_path: str, state: Dict[str, Any], log) -> Dict[str, Any]:
    """Fill ``state`` from progress.json and partial_online_results.json in ``resume_path``."""
    progress_file = os.path.join(resume_path, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
        log.info(f"Resuming from progress: {progress}")
        if progress.get("mode") == "offline":
            state["start_epoch"] = progress.get("epoch", 1)
            state["start_step"] = progress.get("step", 0)
        elif progress.get("mode") == "online":
            # Resume at the NEXT window after the last completed one
            state["start_window"] = progress.get("window", 0) + 1
            state["start_global_step"] = progress.get("global_step", 0)

    partial_file = os.path.join(resume_path, "partial_online_results.json")
    if os.path.exists(partial_file):
        with open(partial_file) as f:
            state["prior_online_results"] = json.load(f)
    return state
