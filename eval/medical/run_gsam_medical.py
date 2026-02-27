#!/usr/bin/env python3
"""
run_gsam_medical.py
===================
Entry point for GSAM medical NER experiments (BC5CDR benchmark).

This mirrors eval/finance/run_gsam.py but uses the medical DataProcessor
and the BC5CDR ontology instead of the XBRL taxonomy.

Usage
-----
# First, download the data:
python eval/medical/data/download_bc5cdr.py

# Then run GSAM in online mode:
python -m eval.medical.run_gsam_medical \
    --task_name bc5cdr \
    --mode online \
    --api_provider sambanova \
    --generator_model DeepSeek-V3.1 \
    --save_path ./results/bc5cdr_online

# Or offline mode (multi-epoch):
python -m eval.medical.run_gsam_medical \
    --task_name bc5cdr \
    --mode offline \
    --num_epochs 3 \
    --api_provider sambanova \
    --save_path ./results/bc5cdr_offline

# Smoke test (5 samples):
python -m eval.medical.run_gsam_medical \
    --task_name bc5cdr \
    --mode online \
    --api_provider sambanova \
    --max_samples 5 \
    --save_path ./results/bc5cdr_smoke

Structural Analogy to FiNER
----------------------------
Component           FiNER (Finance)             BC5CDR (Medical)
-----------------   --------------------------  --------------------------
Task                XBRL entity recognition     Medical entity recognition
Entity types        139 XBRL tags               Chemical, Disease (+subtypes)
Formal ontology     XBRL / US-GAAP taxonomy     MeSH + SNOMED CT hierarchy
Ontology file       xbrl_taxonomy.json          bc5cdr_ontology.json
Data format         FinLoRA batched JSONL        GSAM medical JSONL
Answer format       XBRL concept name           "Chemical" or "Disease"
Evaluation          Exact-match accuracy        Exact-match accuracy
Transfer benchmark  FiNER-Transfer              MedNER-Transfer (future work)
"""

import os
import sys
import json
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure repo root is on the path when running as a module
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval.medical.data_processor import MedicalDataProcessor
from gsam import GSAM


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="GSAM Medical NER Experiments (BC5CDR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task
    parser.add_argument("--task_name", type=str, default="bc5cdr",
                        choices=["bc5cdr"],
                        help="Medical NER task name")
    parser.add_argument("--mode", type=str, default="online",
                        choices=["offline", "online", "eval_only"])

    # API / models
    parser.add_argument("--api_provider", type=str, default="sambanova",
                        choices=["sambanova", "together", "openai", "modal"])
    parser.add_argument("--generator_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--reflector_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--curator_model", type=str, default="DeepSeek-V3.1")

    # Training
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--online_eval_frequency", type=int, default=15)
    parser.add_argument("--save_steps", type=int, default=50)

    # System
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--playbook_token_budget", type=int, default=80000)
    parser.add_argument("--test_workers", type=int, default=20)

    # Output
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")

    # GSAM
    parser.add_argument("--ontology_path", type=str,
                        default="./eval/medical/data/bc5cdr_ontology.json")
    parser.add_argument("--merge_threshold", type=float, default=0.9)
    parser.add_argument("--retrieval_depth", type=int, default=2)
    parser.add_argument("--prune_frequency", type=int, default=50)

    # Ablations
    parser.add_argument("--no_ontology", action="store_true")
    parser.add_argument("--no_failure_cascades", action="store_true")
    parser.add_argument("--embedding_only_retrieval", action="store_true")
    parser.add_argument("--untyped_edges", action="store_true")
    parser.add_argument("--no_multi_epoch_refinement", action="store_true")

    # Smoke test
    parser.add_argument("--max_samples", type=int, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run the download script first:\n"
            "    python eval/medical/data/download_bc5cdr.py"
        )
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return data


def preprocess(task_name, config, mode, max_samples=None):
    processor = MedicalDataProcessor(task_name=task_name)

    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None
        test_samples = processor.process_task_data(load_jsonl(config["test_data"]))
        if max_samples:
            test_samples = test_samples[:max_samples]
    else:
        train_samples = processor.process_task_data(load_jsonl(config["train_data"]))
        val_samples = processor.process_task_data(load_jsonl(config["val_data"]))
        if max_samples:
            train_samples = train_samples[:max_samples]
            val_samples = val_samples[:max(max_samples // 5, 5)]

        test_samples = []
        if "test_data" in config:
            test_samples = processor.process_task_data(load_jsonl(config["test_data"]))
            if max_samples:
                test_samples = test_samples[:max_samples]

    return train_samples, val_samples, test_samples, processor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"GSAM MEDICAL NER — {args.task_name.upper()}")
    print(f"{'='*60}")
    print(f"Task        : {args.task_name}")
    print(f"Mode        : {args.mode.upper()}")
    print(f"API         : {args.api_provider}")
    print(f"Generator   : {args.generator_model}")
    print(f"Ontology    : {'disabled' if args.no_ontology else args.ontology_path}")
    print(f"{'='*60}\n")

    # Load data config
    config_path = "./eval/medical/data/sample_config.json"
    with open(config_path) as f:
        task_config = json.load(f)

    train_samples, val_samples, test_samples, data_processor = preprocess(
        args.task_name,
        task_config[args.task_name],
        args.mode,
        args.max_samples,
    )

    # Initialise GSAM with the medical ontology
    gsam_system = GSAM(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        # Pass the medical ontology path here; the ontology loader will
        # build Concept nodes + is_a edges from bc5cdr_ontology.json.
        taxonomy_path=args.ontology_path,
        formula_data_path=None,
        merge_threshold=args.merge_threshold,
        retrieval_depth=args.retrieval_depth,
        prune_frequency=args.prune_frequency,
        no_ontology=args.no_ontology,
        no_failure_cascades=args.no_failure_cascades,
        embedding_only_retrieval=args.embedding_only_retrieval,
        untyped_edges=args.untyped_edges,
    )

    run_config = {
        "num_epochs": args.num_epochs,
        "max_num_rounds": args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "eval_steps": args.eval_steps,
        "online_eval_frequency": args.online_eval_frequency,
        "save_steps": args.save_steps,
        "playbook_token_budget": args.playbook_token_budget,
        "task_name": args.task_name,
        "mode": args.mode,
        "json_mode": args.json_mode,
        "no_ground_truth": args.no_ground_truth,
        "save_dir": args.save_path,
        "test_workers": args.test_workers,
        "api_provider": args.api_provider,
        "ontology_path": args.ontology_path,
        "no_multi_epoch_refinement": args.no_multi_epoch_refinement,
    }

    results = gsam_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=run_config,
    )

    print(f"\nResults: {results}")


if __name__ == "__main__":
    main()
