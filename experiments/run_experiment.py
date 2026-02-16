#!/usr/bin/env python3
"""
GSAM Experiment Runner

Reads a JSON config file and runs a GSAM or ACE experiment.
Supports running single experiments or ablation suites.

Usage:
    # Single experiment
    python -m experiments.run_experiment --config experiments/configs/gsam_finer_online.json --save_path results

    # Ablation suite (runs all configs in a directory)
    python -m experiments.run_experiment --config_dir experiments/configs/ --save_path results --filter ablation
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment config from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def build_command(config: Dict[str, Any], save_path: str) -> List[str]:
    """Build the command line invocation from config."""
    system = config.get("system", "gsam")

    if system == "gsam":
        module = "eval.finance.run_gsam"
    else:
        module = "eval.finance.run"

    cmd = [sys.executable, "-m", module]

    # Required args
    cmd.extend(["--task_name", config["task_name"]])
    cmd.extend(["--mode", config["mode"]])
    cmd.extend(["--save_path", save_path])

    # Model args
    cmd.extend(["--api_provider", config.get("api_provider", "sambanova")])
    cmd.extend(["--generator_model", config.get("generator_model", "DeepSeek-V3.1")])
    cmd.extend(["--reflector_model", config.get("reflector_model", "DeepSeek-V3.1")])
    cmd.extend(["--curator_model", config.get("curator_model", "DeepSeek-V3.1")])

    # Training args
    cmd.extend(["--num_epochs", str(config.get("num_epochs", 1))])
    cmd.extend(["--max_num_rounds", str(config.get("max_num_rounds", 3))])
    cmd.extend(["--curator_frequency", str(config.get("curator_frequency", 1))])
    cmd.extend(["--eval_steps", str(config.get("eval_steps", 100))])
    cmd.extend(["--save_steps", str(config.get("save_steps", 50))])
    cmd.extend(["--max_tokens", str(config.get("max_tokens", 4096))])
    cmd.extend(["--playbook_token_budget", str(config.get("playbook_token_budget", 80000))])
    cmd.extend(["--test_workers", str(config.get("test_workers", 20))])

    if "online_eval_frequency" in config:
        cmd.extend(["--online_eval_frequency", str(config["online_eval_frequency"])])

    # Boolean flags
    if config.get("json_mode"):
        cmd.append("--json_mode")
    if config.get("no_ground_truth"):
        cmd.append("--no_ground_truth")

    # GSAM-specific args
    if system == "gsam":
        if "taxonomy_path" in config:
            cmd.extend(["--taxonomy_path", config["taxonomy_path"]])
        if "merge_threshold" in config:
            cmd.extend(["--merge_threshold", str(config["merge_threshold"])])
        if "retrieval_depth" in config:
            cmd.extend(["--retrieval_depth", str(config["retrieval_depth"])])
        if "prune_frequency" in config:
            cmd.extend(["--prune_frequency", str(config["prune_frequency"])])
        if config.get("no_ontology"):
            cmd.append("--no_ontology")
        if config.get("no_failure_cascades"):
            cmd.append("--no_failure_cascades")
        if config.get("embedding_only_retrieval"):
            cmd.append("--embedding_only_retrieval")
        if config.get("untyped_edges"):
            cmd.append("--untyped_edges")
        if config.get("max_samples"):
            cmd.extend(["--max_samples", str(config["max_samples"])])

    # ACE-specific args
    else:
        if config.get("initial_playbook_path"):
            cmd.extend(["--initial_playbook_path", config["initial_playbook_path"]])
        if config.get("use_bulletpoint_analyzer"):
            cmd.append("--use_bulletpoint_analyzer")

    return cmd


def run_single_experiment(config_path: str, save_path: str) -> None:
    """Run a single experiment from a config file."""
    config = load_config(config_path)
    exp_name = config.get("experiment_name", os.path.basename(config_path).replace(".json", ""))

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"{'='*60}")

    exp_save = os.path.join(save_path, exp_name)
    os.makedirs(exp_save, exist_ok=True)

    # Save config copy
    with open(os.path.join(exp_save, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    cmd = build_command(config, exp_save)
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\nExperiment {exp_name} failed with return code {result.returncode}")
    else:
        print(f"\nExperiment {exp_name} completed successfully")


def run_suite(config_dir: str, save_path: str, name_filter: Optional[str] = None) -> None:
    """Run all experiments in a config directory."""
    config_files = sorted([
        f for f in os.listdir(config_dir)
        if f.endswith(".json")
    ])

    if name_filter:
        config_files = [f for f in config_files if name_filter in f]

    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUITE: {len(config_files)} experiments")
    if name_filter:
        print(f"Filter: '{name_filter}'")
    print(f"{'='*60}")

    for i, config_file in enumerate(config_files, 1):
        print(f"\n[{i}/{len(config_files)}] {config_file}")
        config_path = os.path.join(config_dir, config_file)
        run_single_experiment(config_path, save_path)

    print(f"\n{'='*60}")
    print(f"SUITE COMPLETE: {len(config_files)} experiments")
    print(f"Results saved to: {save_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Run GSAM experiments")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to single experiment config JSON")
    parser.add_argument("--config_dir", type=str, default=None,
                        help="Path to directory of config JSONs (runs all)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Base directory to save results")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run configs whose name contains this string")
    args = parser.parse_args()

    if args.config:
        run_single_experiment(args.config, args.save_path)
    elif args.config_dir:
        run_suite(args.config_dir, args.save_path, args.filter)
    else:
        parser.error("Provide either --config or --config_dir")


if __name__ == "__main__":
    main()
