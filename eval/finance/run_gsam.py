#!/usr/bin/env python3
"""
GSAM Entry Point for Financial Experiments

Mirrors eval/finance/run.py but initializes GSAM instead of ACE.
Supports the same CLI arguments plus GSAM-specific options.
"""
import os
import json
import argparse
from datetime import datetime
from .data_processor import DataProcessor

from gsam import GSAM


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GSAM System - Financial Experiments')

    # Task configuration
    parser.add_argument("--task_name", type=str, required=True,
                        help="Name of the task (e.g., 'finer', 'formula')")
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"],
                        help="Run mode")

    # Model configuration
    parser.add_argument("--api_provider", type=str, default="sambanova",
                        choices=["sambanova", "together", "openai", "modal"])
    parser.add_argument("--generator_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--reflector_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--curator_model", type=str, default="DeepSeek-V3.1")

    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs (default 5 for offline per paper §6.4, "
                             "online always uses 1)")
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--online_eval_frequency", type=int, default=15)
    parser.add_argument("--save_steps", type=int, default=50)

    # System configuration
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--playbook_token_budget", type=int, default=80000)
    parser.add_argument("--test_workers", type=int, default=20)

    # Prompt configuration
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")

    # Output configuration
    parser.add_argument("--save_path", type=str, required=True)

    # GSAM-specific arguments
    parser.add_argument("--taxonomy_path", type=str,
                        default="./eval/finance/data/xbrl_taxonomy.json",
                        help="Path to XBRL taxonomy JSON")
    parser.add_argument("--merge_threshold", type=float, default=0.9,
                        help="Cosine similarity threshold for node deduplication")
    parser.add_argument("--retrieval_depth", type=int, default=2,
                        help="BFS depth for graph retrieval")
    parser.add_argument("--prune_frequency", type=int, default=50,
                        help="Prune low-utility nodes every N steps")

    # Ablation flags
    parser.add_argument("--no_ontology", action="store_true",
                        help="Ablation: skip ontology initialization")
    parser.add_argument("--no_failure_cascades", action="store_true",
                        help="Ablation: skip anti-pattern creation and failure edges")
    parser.add_argument("--embedding_only_retrieval", action="store_true",
                        help="Ablation: use embedding-only retrieval (no graph BFS)")
    parser.add_argument("--untyped_edges", action="store_true",
                        help="Ablation: all edges are generic 'related_to'")
    parser.add_argument("--no_multi_epoch_refinement", action="store_true",
                        help="Ablation: disable consolidate_epoch() in offline mode")

    # Limit for smoke testing
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for smoke testing)")

    return parser.parse_args()


def load_data(data_path: str):
    """Load data from JSONL file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data


def preprocess_data(task_name, config, mode, max_samples=None):
    """Load and preprocess data."""
    processor = DataProcessor(task_name=task_name)

    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None
        test_samples = load_data(config["test_data"])
        test_samples = processor.process_task_data(test_samples)
        if max_samples:
            test_samples = test_samples[:max_samples]
    else:
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)
        if max_samples:
            train_samples = train_samples[:max_samples]
            val_samples = val_samples[:max(max_samples // 5, 5)]

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
            if max_samples:
                test_samples = test_samples[:max_samples]
        else:
            test_samples = []

    return train_samples, val_samples, test_samples, processor


def main():
    """Main execution function."""
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"GSAM SYSTEM")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Generator Model: {args.generator_model}")
    print(f"Ontology: {'Disabled' if args.no_ontology else args.taxonomy_path}")
    print(f"Failure Cascades: {'Disabled' if args.no_failure_cascades else 'Enabled'}")
    print(f"Retrieval: {'Embedding-only' if args.embedding_only_retrieval else 'Graph BFS'}")
    print(f"Edge Types: {'Untyped' if args.untyped_edges else 'Typed'}")
    print(f"{'='*60}\n")

    # Load task config
    with open("./eval/finance/data/sample_config.json", 'r') as f:
        task_config = json.load(f)

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name,
        task_config[args.task_name],
        args.mode,
        args.max_samples,
    )

    # Determine formula data path for ontology
    formula_data_path = None
    if args.task_name == "formula" or "formula" in task_config:
        formula_data_path = task_config.get("formula", {}).get("train_data")

    # Create GSAM system
    gsam_system = GSAM(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        taxonomy_path=args.taxonomy_path,
        formula_data_path=formula_data_path,
        merge_threshold=args.merge_threshold,
        retrieval_depth=args.retrieval_depth,
        prune_frequency=args.prune_frequency,
        no_ontology=args.no_ontology,
        no_failure_cascades=args.no_failure_cascades,
        embedding_only_retrieval=args.embedding_only_retrieval,
        untyped_edges=args.untyped_edges,
    )

    # Prepare config
    config = {
        'num_epochs': args.num_epochs,
        'max_num_rounds': args.max_num_rounds,
        'curator_frequency': args.curator_frequency,
        'eval_steps': args.eval_steps,
        'online_eval_frequency': args.online_eval_frequency,
        'save_steps': args.save_steps,
        'playbook_token_budget': args.playbook_token_budget,
        'task_name': args.task_name,
        'mode': args.mode,
        'json_mode': args.json_mode,
        'no_ground_truth': args.no_ground_truth,
        'save_dir': args.save_path,
        'test_workers': args.test_workers,
        'api_provider': args.api_provider,
        # GSAM-specific
        'taxonomy_path': args.taxonomy_path,
        'merge_threshold': args.merge_threshold,
        'retrieval_depth': args.retrieval_depth,
        'prune_frequency': args.prune_frequency,
        'no_ontology': args.no_ontology,
        'no_failure_cascades': args.no_failure_cascades,
        'embedding_only_retrieval': args.embedding_only_retrieval,
        'untyped_edges': args.untyped_edges,
        'no_multi_epoch_refinement': args.no_multi_epoch_refinement,
    }

    # Run
    results = gsam_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=config,
    )


if __name__ == "__main__":
    main()
