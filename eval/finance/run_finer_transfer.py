#!/usr/bin/env python3
"""
run_finer_transfer.py
=====================
Entry point for the FiNER-Transfer cross-concept transfer benchmark.

Tests Hypothesis H1 from the GSAM paper: "Graph-structured memory with
ontology grounding improves cross-concept transfer compared to flat storage."

Usage
-----
# Full benchmark (GSAM vs ACE)
python -m eval.finance.run_finer_transfer \
    --method gsam \
    --api_provider sambanova \
    --generator_model DeepSeek-V3.1 \
    --save_path ./results/finer_transfer

# Smoke test: one sibling pair, limit examples
python -m eval.finance.run_finer_transfer \
    --method gsam \
    --api_provider sambanova \
    --max_pairs 1 \
    --max_examples_per_concept 5 \
    --save_path ./results/finer_transfer_smoke

Design
------
For each concept pair (A, B):
  1. Create a *fresh* system (no prior adaptation) via system_factory().
  2. Evaluate baseline accuracy on B.
  3. Adapt on A examples (graph/playbook grows).
  4. Evaluate transfer accuracy on B.
  5. Δ_transfer = transfer - baseline.

Creating a fresh system per experiment is the key correctness requirement:
it prevents knowledge learned for pair (A,B) from contaminating the baseline
or adaptation of pair (C,D).  See finer_transfer.py::evaluate_transfer for
the detailed protocol.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Callable

# ---------------------------------------------------------------------------
# Ensure repo root is on the path when running as a module
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval.finance.data_processor import DataProcessor
from eval.finance.finer_transfer import (
    load_taxonomy,
    build_concept_pairs,
    build_transfer_splits,
    evaluate_transfer,
    compute_aggregate_transfer_metrics,
    save_concept_pairs,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="FiNER-Transfer Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Method
    parser.add_argument("--method", type=str, default="gsam",
                        choices=["gsam", "ace"],
                        help="Memory system to evaluate")

    # API / models
    parser.add_argument("--api_provider", type=str, default="sambanova",
                        choices=["sambanova", "together", "openai", "modal"])
    parser.add_argument("--generator_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--reflector_model", type=str, default="DeepSeek-V3.1")
    parser.add_argument("--curator_model", type=str, default="DeepSeek-V3.1")

    # Data paths
    parser.add_argument("--taxonomy_path", type=str,
                        default="./eval/finance/data/xbrl_taxonomy.json")
    parser.add_argument("--finer_data_path", type=str,
                        default="./eval/finance/data/finer_train_batched_1000_samples.jsonl")

    # Experiment scope
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Limit number of concept pairs (None = run all)")
    parser.add_argument("--only_siblings", action="store_true",
                        help="Only run sibling (near-transfer) pairs")
    parser.add_argument("--max_examples_per_concept", type=int, default=None,
                        help="Cap examples per concept for faster runs")
    parser.add_argument("--min_examples_per_concept", type=int, default=3,
                        help="Skip pairs with fewer examples than this")

    # GSAM-specific knobs
    parser.add_argument("--merge_threshold", type=float, default=0.9)
    parser.add_argument("--retrieval_depth", type=int, default=2)
    parser.add_argument("--prune_frequency", type=int, default=50)
    parser.add_argument("--no_ontology", action="store_true")
    parser.add_argument("--no_failure_cascades", action="store_true")
    parser.add_argument("--embedding_only_retrieval", action="store_true")
    parser.add_argument("--untyped_edges", action="store_true")

    # Training config shared across per-experiment systems
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--playbook_token_budget", type=int, default=80000)
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")

    # Output
    parser.add_argument("--save_path", type=str, required=True)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# System factory helpers
# ---------------------------------------------------------------------------


def _make_gsam_factory(args, taxonomy_path: str) -> Callable:
    """
    Return a zero-argument callable that creates a fresh GSAM system.

    Each call to the factory produces an isolated instance with only ontology
    nodes pre-loaded and zero experiential knowledge.
    """
    from gsam import GSAM

    def factory():
        return GSAM(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            reflector_model=args.reflector_model,
            curator_model=args.curator_model,
            max_tokens=args.max_tokens,
            taxonomy_path=taxonomy_path,
            formula_data_path=None,
            merge_threshold=args.merge_threshold,
            retrieval_depth=args.retrieval_depth,
            prune_frequency=args.prune_frequency,
            no_ontology=args.no_ontology,
            no_failure_cascades=args.no_failure_cascades,
            embedding_only_retrieval=args.embedding_only_retrieval,
            untyped_edges=args.untyped_edges,
        )

    return factory


def _make_ace_factory(args) -> Callable:
    """
    Return a zero-argument callable that creates a fresh ACE system.
    """
    from ace.ace import ACE  # type: ignore[import]

    def factory():
        return ACE(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            reflector_model=args.reflector_model,
            curator_model=args.curator_model,
            max_tokens=args.max_tokens,
        )

    return factory


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_finer_data(path: str, processor: DataProcessor):
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(__import__("json").loads(line))
    return processor.process_task_data(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_path, f"finer_transfer_{args.method}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'='*60}")
    print(f"FiNER-Transfer Benchmark")
    print(f"  Method       : {args.method.upper()}")
    print(f"  API provider : {args.api_provider}")
    print(f"  Results dir  : {run_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Build concept pairs from XBRL taxonomy
    # ------------------------------------------------------------------
    taxonomy = load_taxonomy(args.taxonomy_path)
    pairs = build_concept_pairs(taxonomy)

    # Filter to siblings only if requested
    if args.only_siblings:
        pairs = [p for p in pairs if p["pair_type"] == "sibling"]
        print(f"Filtered to {len(pairs)} sibling pairs")

    # Limit pairs for smoke testing
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
        print(f"Limited to {args.max_pairs} pairs")

    save_concept_pairs(pairs, os.path.join(run_dir, "concept_pairs.json"))

    # ------------------------------------------------------------------
    # Load FiNER data and build transfer splits
    # ------------------------------------------------------------------
    processor = DataProcessor(task_name="finer")
    finer_data = _load_finer_data(args.finer_data_path, processor)

    experiments = build_transfer_splits(
        finer_data, pairs,
        min_examples_per_concept=args.min_examples_per_concept,
    )

    if args.max_examples_per_concept is not None:
        for exp in experiments:
            exp["source_examples"] = exp["source_examples"][:args.max_examples_per_concept]
            exp["target_examples"] = exp["target_examples"][:args.max_examples_per_concept]

    print(f"\nRunning {len(experiments)} transfer experiments ...\n")

    # ------------------------------------------------------------------
    # Build system factory
    # ------------------------------------------------------------------
    if args.method == "gsam":
        factory = _make_gsam_factory(args, args.taxonomy_path)
    else:
        factory = _make_ace_factory(args)

    # ------------------------------------------------------------------
    # Run transfer experiments
    # ------------------------------------------------------------------
    run_config = {
        "max_num_rounds": args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "playbook_token_budget": args.playbook_token_budget,
        "json_mode": args.json_mode,
        "no_ground_truth": args.no_ground_truth,
    }

    results = []
    for idx, experiment in enumerate(experiments):
        print(f"\n[{idx+1}/{len(experiments)}] "
              f"{experiment['source_concept']} -> {experiment['target_concept']} "
              f"({experiment['pair']['pair_type']})")

        result = evaluate_transfer(
            method_name=args.method,
            experiment=experiment,
            system_factory=factory,
            data_processor=processor,
            config=run_config,
            save_path=run_dir,
        )
        results.append(result)

        # Save incremental results after every experiment
        with open(os.path.join(run_dir, "transfer_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    aggregate = compute_aggregate_transfer_metrics(results)

    print(f"\n{'='*60}")
    print(f"FiNER-Transfer Results ({args.method.upper()})")
    print(f"{'='*60}")
    print(f"  Total pairs         : {aggregate['total_pairs']}")
    print(f"  Sibling pairs       : {aggregate['sibling_pairs']}")
    print(f"  Distant pairs       : {aggregate['distant_pairs']}")
    print(f"  Near-transfer rate  : {aggregate['near_transfer_rate']:.3f}")
    print(f"  Far-transfer rate   : {aggregate['far_transfer_rate']:.3f}")
    print(f"  Transfer precision  : {aggregate['transfer_precision']:.3f}")
    print(f"  Negative-xfer rate  : {aggregate['negative_transfer_rate']:.3f}")
    print(f"{'='*60}\n")

    final_output = {"aggregate": aggregate, "per_pair_results": results}
    with open(os.path.join(run_dir, "final_results.json"), "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
