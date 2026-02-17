#!/usr/bin/env python3
"""
FiNER-Transfer Benchmark

A controlled benchmark for measuring cross-concept transfer
within the XBRL taxonomy. Tests Hypothesis H1: whether graph-structured
memory with ontology grounding improves cross-concept transfer.

Construction:
1. Group 139 FiNER entity types by XBRL taxonomy position
2. Identify sibling concept pairs (share a direct parent)
3. For each pair, create **non-overlapping** source/target splits
4. Measure transfer: adapt on source, evaluate on target

Key design decisions vs. the naive approach:
- Each FiNER sample is a "batch" of 4 questions with 4 comma-separated
  target tags. A sample is assigned to concept C only if C appears in
  its target AND the sibling concept does NOT appear in the same sample.
  This eliminates data contamination (31.5% of raw examples).
- Transfer accuracy is computed per-tag-position: only the positions
  in the target that correspond to the tested concept count.
- Each transfer pair gets a fresh system to prevent cross-pair leakage.
"""

import os
import json
import copy
import random
import argparse
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict

from .data_processor import DataProcessor


# ------------------------------------------------------------------ #
# Taxonomy & Pair Construction
# ------------------------------------------------------------------ #

def load_taxonomy(taxonomy_path: str) -> Dict:
    """Load the XBRL taxonomy."""
    with open(taxonomy_path) as f:
        return json.load(f)


def build_concept_pairs(taxonomy: Dict) -> List[Dict]:
    """
    Build sibling concept pairs from the XBRL taxonomy.

    Sibling concepts share a direct parent (subcategory) in the taxonomy.
    These pairs are used for near-transfer experiments.

    Also generates distant pairs (cross-category) as negative controls.

    Returns:
        List of pair dicts.
    """
    pairs = []
    categories = taxonomy.get("categories", {})

    # Collect all entities organized by subcategory
    subcat_entities = {}  # (category, subcat) -> [entity_names]

    for cat_name, cat_data in categories.items():
        for subcat_name, subcat_data in cat_data.get("children", {}).items():
            entities = subcat_data.get("entities", [])
            if len(entities) >= 2:
                subcat_entities[(cat_name, subcat_name)] = entities

    # Generate sibling pairs (within same subcategory)
    for (cat, subcat), entities in subcat_entities.items():
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                pairs.append({
                    "concept_a": entities[i],
                    "concept_b": entities[j],
                    "parent": subcat,
                    "category": cat,
                    "pair_type": "sibling",
                })

    # Generate distant pairs (across different categories) as negative controls
    all_entities = []
    for entities in subcat_entities.values():
        all_entities.extend(entities)

    entity_to_cat = {}
    for (cat, subcat), entities in subcat_entities.items():
        for e in entities:
            entity_to_cat[e] = cat

    random.seed(42)
    distant_pairs = []
    attempts = 0
    while len(distant_pairs) < min(20, len(pairs)) and attempts < 1000:
        a = random.choice(all_entities)
        b = random.choice(all_entities)
        if a != b and entity_to_cat.get(a) != entity_to_cat.get(b):
            distant_pairs.append({
                "concept_a": a,
                "concept_b": b,
                "parent": "N/A",
                "category": "cross-category",
                "pair_type": "distant",
            })
        attempts += 1

    return pairs + distant_pairs


# ------------------------------------------------------------------ #
# Data Split Construction — contamination-free
# ------------------------------------------------------------------ #

def _get_tags(example: Dict) -> List[str]:
    """Extract the list of tags from a target string."""
    return [t.strip() for t in example.get("target", "").split(",") if t.strip()]


def build_transfer_splits(
    finer_data: List[Dict],
    concept_pairs: List[Dict],
    min_examples_per_concept: int = 3,
) -> List[Dict]:
    """
    Build source/target splits for each concept pair,
    **excluding contaminated examples**.

    An example is contaminated for pair (A, B) if its target contains
    BOTH A and B. Such examples would leak target-concept knowledge
    into the source adaptation phase, invalidating the transfer
    measurement.

    Args:
        finer_data: Processed FiNER data.
        concept_pairs: List of concept pair dicts.
        min_examples_per_concept: Minimum examples needed per concept.

    Returns:
        List of transfer experiment configs.
    """
    # Index examples by concept
    concept_to_examples = defaultdict(list)
    for example in finer_data:
        for tag in _get_tags(example):
            concept_to_examples[tag].append(example)

    # Build transfer experiments with contamination filtering
    experiments = []
    skipped_contamination = 0
    skipped_insufficient = 0

    for pair in concept_pairs:
        concept_a = pair["concept_a"]
        concept_b = pair["concept_b"]

        # Filter: source examples must mention A but NOT B
        source_clean = [
            ex for ex in concept_to_examples.get(concept_a, [])
            if concept_b not in _get_tags(ex)
        ]
        # Filter: target examples must mention B but NOT A
        target_clean = [
            ex for ex in concept_to_examples.get(concept_b, [])
            if concept_a not in _get_tags(ex)
        ]

        raw_a = len(concept_to_examples.get(concept_a, []))
        raw_b = len(concept_to_examples.get(concept_b, []))

        if (len(source_clean) < min_examples_per_concept
                or len(target_clean) < min_examples_per_concept):
            if raw_a >= min_examples_per_concept and raw_b >= min_examples_per_concept:
                skipped_contamination += 1
            else:
                skipped_insufficient += 1
            continue

        experiments.append({
            "pair": pair,
            "source_concept": concept_a,
            "target_concept": concept_b,
            "source_examples": source_clean,
            "target_examples": target_clean,
            "source_count": len(source_clean),
            "target_count": len(target_clean),
            "source_raw_count": raw_a,
            "target_raw_count": raw_b,
        })

    sibling_count = sum(1 for e in experiments if e["pair"]["pair_type"] == "sibling")
    distant_count = sum(1 for e in experiments if e["pair"]["pair_type"] == "distant")
    print(f"Built {len(experiments)} transfer experiments "
          f"({sibling_count} sibling, {distant_count} distant)")
    if skipped_contamination:
        print(f"  Skipped {skipped_contamination} pairs due to cross-contamination")
    if skipped_insufficient:
        print(f"  Skipped {skipped_insufficient} pairs due to insufficient data")

    return experiments


# ------------------------------------------------------------------ #
# Concept-Specific Accuracy
# ------------------------------------------------------------------ #

def concept_specific_accuracy(
    answer: str,
    target: str,
    concept: str,
) -> Tuple[int, int]:
    """
    Compute accuracy only on the tag positions relevant to *concept*.

    FiNER targets are comma-separated (e.g. "A,B,A,C"). If we are
    testing transfer to concept A, only positions where the ground truth
    is A should count.

    Returns:
        (correct_positions, total_positions)
    """
    pred_tags = [t.strip() for t in answer.split(",")]
    true_tags = [t.strip() for t in target.split(",")]

    correct = 0
    total = 0
    for i, true_tag in enumerate(true_tags):
        if true_tag == concept:
            total += 1
            if i < len(pred_tags) and pred_tags[i] == true_tag:
                correct += 1

    return correct, total


# ------------------------------------------------------------------ #
# Transfer Evaluation
# ------------------------------------------------------------------ #

def evaluate_transfer(
    method_name: str,
    experiment: Dict,
    system_factory: Callable,
    data_processor: DataProcessor,
    config: Dict,
    save_path: str,
) -> Dict[str, Any]:
    """
    Run a single transfer experiment with a **fresh system**.

    Protocol:
    1. Baseline: Evaluate on target with a fresh (unadapted) system
    2. Adapt: Train the same fresh system on source examples
    3. Transfer: Evaluate on target with adapted system

    Using a fresh system per pair prevents knowledge leakage from
    previous experiments.

    Args:
        method_name: "ace" or "gsam"
        experiment: Transfer experiment config.
        system_factory: Callable that returns a fresh system instance.
        data_processor: DataProcessor instance.
        config: Run configuration.
        save_path: Path to save results.

    Returns:
        Dict with transfer metrics.
    """
    source_examples = experiment["source_examples"]
    target_examples = experiment["target_examples"]
    source_concept = experiment["source_concept"]
    target_concept = experiment["target_concept"]

    print(f"\nTransfer: {source_concept} -> {target_concept}")
    print(f"  Source examples (clean): {len(source_examples)}")
    print(f"  Target examples (clean): {len(target_examples)}")

    # Create a fresh system for this experiment
    system = system_factory()

    from utils import extract_answer

    # ---- Step 1: Baseline on target (no adaptation) ----
    baseline_correct = 0
    baseline_total = 0
    for example in target_examples:
        target = example.get("target", "")
        question = example.get("question", example.get("context", ""))
        context = example.get("context", "")

        try:
            if hasattr(system, 'knowledge_graph'):
                ctx, _ = system.graph_retriever.retrieve(question, context)
                resp, _, _ = system.generator.generate(
                    question=question, graph_context=ctx, context=context)
            else:
                resp, _, _ = system.generator.generate(
                    question=question,
                    playbook=getattr(system, 'playbook', ''),
                    context=context)
            answer = extract_answer(resp)
            c, t = concept_specific_accuracy(answer, target, target_concept)
            baseline_correct += c
            baseline_total += t
        except Exception as e:
            print(f"  Warning: baseline eval failed: {e}")

    baseline_accuracy = baseline_correct / baseline_total if baseline_total > 0 else 0

    # ---- Step 2: Adapt on source examples ----
    config_params = {
        'max_num_rounds': config.get('max_num_rounds', 3),
        'curator_frequency': config.get('curator_frequency', 1),
        'token_budget': config.get('playbook_token_budget', 80000),
        'use_json_mode': config.get('json_mode', False),
        'no_ground_truth': config.get('no_ground_truth', False),
    }

    log_dir = os.path.join(save_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    for i, example in enumerate(source_examples):
        print(f"  Adapting on source {i+1}/{len(source_examples)}")
        try:
            if hasattr(system, '_train_single_sample'):
                system._train_single_sample(
                    task_dict=example,
                    data_processor=data_processor,
                    step_id=f"transfer_{source_concept}_s_{i}",
                    step=i + 1,
                    log_dir=log_dir,
                    config_params=config_params,
                    total_samples=len(source_examples),
                )
        except Exception as e:
            print(f"  Warning: Adaptation failed on source example {i}: {e}")

    # ---- Step 3: Evaluate on target after adaptation ----
    transfer_correct = 0
    transfer_total = 0
    for example in target_examples:
        target = example.get("target", "")
        question = example.get("question", example.get("context", ""))
        context = example.get("context", "")

        try:
            if hasattr(system, 'knowledge_graph'):
                ctx, _ = system.graph_retriever.retrieve(question, context)
                resp, _, _ = system.generator.generate(
                    question=question, graph_context=ctx, context=context)
            else:
                resp, _, _ = system.generator.generate(
                    question=question,
                    playbook=getattr(system, 'playbook', ''),
                    context=context)
            answer = extract_answer(resp)
            c, t = concept_specific_accuracy(answer, target, target_concept)
            transfer_correct += c
            transfer_total += t
        except Exception as e:
            print(f"  Warning: Transfer eval failed: {e}")

    transfer_accuracy = transfer_correct / transfer_total if transfer_total > 0 else 0

    # Compute transfer metrics
    from gsam.metrics import compute_transfer_metrics
    metrics = compute_transfer_metrics(
        source_accuracy=0.0,  # Not measured in this protocol
        target_accuracy_with_transfer=transfer_accuracy,
        target_accuracy_without_transfer=baseline_accuracy,
    )

    result = {
        "source_concept": source_concept,
        "target_concept": target_concept,
        "pair_type": experiment["pair"]["pair_type"],
        "parent": experiment["pair"]["parent"],
        "baseline_accuracy": baseline_accuracy,
        "baseline_correct": baseline_correct,
        "baseline_total": baseline_total,
        "transfer_accuracy": transfer_accuracy,
        "transfer_correct": transfer_correct,
        "transfer_total": transfer_total,
        **metrics,
    }

    print(f"  Baseline: {baseline_accuracy:.3f} -> Transfer: {transfer_accuracy:.3f}"
          f" (delta={metrics['delta_transfer']:.3f})")

    return result


# ------------------------------------------------------------------ #
# Aggregation
# ------------------------------------------------------------------ #

def compute_aggregate_transfer_metrics(results: List[Dict]) -> Dict:
    """
    Compute aggregate transfer metrics across all pairs.

    Returns:
        Dict with:
            - near_transfer_rate: Fraction of sibling pairs with positive transfer
            - far_transfer_rate: Fraction of distant pairs with positive transfer
            - transfer_precision: Average delta for positive transfers
            - negative_transfer_rate: Fraction of pairs with negative transfer
    """
    sibling_results = [r for r in results if r.get("pair_type") == "sibling"]
    distant_results = [r for r in results if r.get("pair_type") == "distant"]

    near_positive = sum(1 for r in sibling_results if r.get("positive_transfer", False))
    far_positive = sum(1 for r in distant_results if r.get("positive_transfer", False))
    negative = sum(1 for r in results if r.get("negative_transfer", False))

    positive_deltas = [r["delta_transfer"] for r in results if r.get("positive_transfer")]

    return {
        "near_transfer_rate": near_positive / len(sibling_results) if sibling_results else 0,
        "far_transfer_rate": far_positive / len(distant_results) if distant_results else 0,
        "transfer_precision": sum(positive_deltas) / len(positive_deltas) if positive_deltas else 0,
        "negative_transfer_rate": negative / len(results) if results else 0,
        "total_pairs": len(results),
        "sibling_pairs": len(sibling_results),
        "distant_pairs": len(distant_results),
    }


# ------------------------------------------------------------------ #
# Persistence
# ------------------------------------------------------------------ #

def save_concept_pairs(pairs: List[Dict], output_path: str) -> None:
    """Save concept pairs to JSON."""
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} concept pairs to {output_path}")


# ------------------------------------------------------------------ #
# CLI: Build Benchmark Data
# ------------------------------------------------------------------ #

def main():
    """Build the FiNER-Transfer benchmark data (pairs + splits)."""
    parser = argparse.ArgumentParser(description="Build FiNER-Transfer Benchmark")
    parser.add_argument("--taxonomy_path", type=str,
                        default="./eval/finance/data/xbrl_taxonomy.json")
    parser.add_argument("--finer_data_path", type=str,
                        default="./eval/finance/data/finer_train_batched_1000_samples.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="./eval/finance/data/finer_transfer")
    args = parser.parse_args()

    # Load taxonomy
    taxonomy = load_taxonomy(args.taxonomy_path)

    # Build concept pairs
    pairs = build_concept_pairs(taxonomy)
    print(f"\nFound {len(pairs)} concept pairs")

    # Save pairs
    os.makedirs(args.output_dir, exist_ok=True)
    save_concept_pairs(pairs, os.path.join(args.output_dir, "concept_pairs.json"))

    # Load FiNER data and build transfer splits
    processor = DataProcessor(task_name="finer")
    finer_data = []
    with open(args.finer_data_path) as f:
        for line in f:
            if line.strip():
                finer_data.append(json.loads(line))
    finer_data = processor.process_task_data(finer_data)

    experiments = build_transfer_splits(finer_data, pairs)

    # Save experiments config
    exp_config = []
    for exp in experiments:
        exp_config.append({
            "source_concept": exp["source_concept"],
            "target_concept": exp["target_concept"],
            "pair_type": exp["pair"]["pair_type"],
            "parent": exp["pair"]["parent"],
            "source_count": exp["source_count"],
            "target_count": exp["target_count"],
            "source_raw_count": exp["source_raw_count"],
            "target_raw_count": exp["target_raw_count"],
        })

    with open(os.path.join(args.output_dir, "transfer_experiments.json"), "w") as f:
        json.dump(exp_config, f, indent=2)
    print(f"Saved {len(exp_config)} transfer experiments")


if __name__ == "__main__":
    main()
