#!/usr/bin/env python3
"""
FiNER-Transfer Experiment Runner

Runs the full FiNER-Transfer benchmark for ACE and/or GSAM,
creating a fresh system instance for each concept pair.
"""

import os
import json
import argparse
from datetime import datetime

from .data_processor import DataProcessor
from .finer_transfer import (
    load_taxonomy,
    build_concept_pairs,
    build_transfer_splits,
    evaluate_transfer,
    compute_aggregate_transfer_metrics,
    save_concept_pairs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run FiNER-Transfer Experiments")

    parser.add_argument("--method", type=str, required=True,
                        choices=["ace", "gsam"],
                        help="Which system to evaluate")
    parser.add_argument("--taxonomy_path", type=str,
                        default="./eval/finance/data/xbrl_taxonomy.json")
    parser.add_argument("--finer_data_path", type=str,
                        default="./eval/finance/data/finer_train_batched_1000_samples.jsonl")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory for results")

    # Model configuration
    parser.add_argument("--api_provider", type=str, default="sambanova")
    parser.add_argument("--generator_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--reflector_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--curator_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--max_tokens", type=int, default=4096)

    # Transfer-specific
    parser.add_argument("--min_examples", type=int, default=3,
                        help="Minimum examples per concept for a viable pair")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Limit number of pairs to run (for testing)")
    parser.add_argument("--pair_types", type=str, default="both",
                        choices=["sibling", "distant", "both"],
                        help="Which pair types to evaluate")

    # GSAM-specific
    parser.add_argument("--merge_threshold", type=float, default=0.9)
    parser.add_argument("--retrieval_depth", type=int, default=2)
    parser.add_argument("--prune_frequency", type=int, default=50)

    # Experiment config
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")

    return parser.parse_args()


def make_system_factory(args):
    """
    Return a callable that creates a fresh ACE or GSAM instance.
    Called once per transfer pair to avoid cross-pair knowledge leakage.
    """
    if args.method == "gsam":
        from gsam import GSAM

        def factory():
            return GSAM(
                api_provider=args.api_provider,
                generator_model=args.generator_model,
                reflector_model=args.reflector_model,
                curator_model=args.curator_model,
                max_tokens=args.max_tokens,
                taxonomy_path=args.taxonomy_path,
                merge_threshold=args.merge_threshold,
                retrieval_depth=args.retrieval_depth,
                prune_frequency=args.prune_frequency,
            )
    else:
        from ace.ace import ACE

        def factory():
            return ACE(
                api_provider=args.api_provider,
                generator_model=args.generator_model,
                reflector_model=args.reflector_model,
                curator_model=args.curator_model,
                max_tokens=args.max_tokens,
            )

    return factory


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"FiNER-Transfer Benchmark")
    print(f"{'='*60}")
    print(f"Method: {args.method.upper()}")
    print(f"Model: {args.generator_model}")
    print(f"{'='*60}\n")

    # Setup
    os.makedirs(args.save_path, exist_ok=True)
    data_processor = DataProcessor(task_name="finer")

    # Build benchmark
    taxonomy = load_taxonomy(args.taxonomy_path)
    pairs = build_concept_pairs(taxonomy)

    # Filter by pair type
    if args.pair_types != "both":
        pairs = [p for p in pairs if p["pair_type"] == args.pair_types]

    # Load FiNER data
    finer_data = []
    with open(args.finer_data_path) as f:
        for line in f:
            if line.strip():
                finer_data.append(json.loads(line))
    finer_data = data_processor.process_task_data(finer_data)

    # Build transfer splits (contamination-free)
    experiments = build_transfer_splits(
        finer_data, pairs, min_examples_per_concept=args.min_examples
    )

    if args.max_pairs:
        experiments = experiments[:args.max_pairs]

    # Create system factory
    system_factory = make_system_factory(args)

    # Run config
    config = {
        'max_num_rounds': args.max_num_rounds,
        'curator_frequency': args.curator_frequency,
        'playbook_token_budget': 80000,
        'json_mode': args.json_mode,
        'no_ground_truth': args.no_ground_truth,
    }

    # Run all transfer experiments
    results = []
    for i, exp in enumerate(experiments):
        print(f"\n{'='*40}")
        print(f"Pair {i+1}/{len(experiments)}")
        print(f"{'='*40}")

        pair_save = os.path.join(
            args.save_path, "pairs",
            f"{exp['source_concept']}_to_{exp['target_concept']}"
        )
        os.makedirs(pair_save, exist_ok=True)

        result = evaluate_transfer(
            method_name=args.method,
            experiment=exp,
            system_factory=system_factory,
            data_processor=data_processor,
            config=config,
            save_path=pair_save,
        )
        results.append(result)

        # Save intermediate results
        with open(os.path.join(args.save_path, "results_partial.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Aggregate and save
    aggregate = compute_aggregate_transfer_metrics(results)

    final = {
        "method": args.method,
        "timestamp": datetime.now().isoformat(),
        "aggregate_metrics": aggregate,
        "per_pair_results": results,
    }

    with open(os.path.join(args.save_path, "transfer_results.json"), "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FiNER-Transfer Results ({args.method.upper()})")
    print(f"{'='*60}")
    print(f"Near-transfer rate:     {aggregate['near_transfer_rate']:.3f}")
    print(f"Far-transfer rate:      {aggregate['far_transfer_rate']:.3f}")
    print(f"Transfer precision:     {aggregate['transfer_precision']:.3f}")
    print(f"Negative transfer rate: {aggregate['negative_transfer_rate']:.3f}")
    print(f"Total pairs evaluated:  {aggregate['total_pairs']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
