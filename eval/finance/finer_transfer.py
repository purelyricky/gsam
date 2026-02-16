#!/usr/bin/env python3
"""
FiNER-Transfer Benchmark

A controlled benchmark for measuring cross-concept transfer
within the XBRL taxonomy. Tests Hypothesis H1: whether graph-structured
memory with ontology grounding improves cross-concept transfer.

Construction:
1. Group 139 FiNER entity types by XBRL taxonomy position
2. Identify sibling concept pairs (share a direct parent)
3. For each pair, create source/target splits
4. Measure transfer: adapt on source, evaluate on target
"""

import os
import json
import random
import argparse
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from .data_processor import DataProcessor


def load_taxonomy(taxonomy_path: str) -> Dict:
    """Load the XBRL taxonomy."""
    with open(taxonomy_path) as f:
        return json.load(f)


def build_concept_pairs(taxonomy: Dict) -> List[Dict]:
    """
    Build sibling concept pairs from the XBRL taxonomy.

    Sibling concepts share a direct parent (subcategory) in the taxonomy.
    These pairs are used for near-transfer experiments.

    Returns:
        List of pair dicts with:
            - concept_a: entity name
            - concept_b: entity name
            - parent: subcategory name
            - category: top-level category name
            - pair_type: "sibling" or "distant"
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

    # Sample distant pairs
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


def build_transfer_splits(
    finer_data: List[Dict],
    concept_pairs: List[Dict],
    min_examples_per_concept: int = 3,
) -> List[Dict]:
    """
    Build source/target splits for each concept pair.

    For each pair (A, B):
    - source_set: Examples involving concept A
    - target_set: Examples involving concept B

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
        target = example.get("target", "")
        for tag in target.split(","):
            tag = tag.strip()
            if tag:
                concept_to_examples[tag].append(example)

    # Build transfer experiments
    experiments = []
    for pair in concept_pairs:
        a_examples = concept_to_examples.get(pair["concept_a"], [])
        b_examples = concept_to_examples.get(pair["concept_b"], [])

        if (len(a_examples) >= min_examples_per_concept
                and len(b_examples) >= min_examples_per_concept):
            experiments.append({
                "pair": pair,
                "source_concept": pair["concept_a"],
                "target_concept": pair["concept_b"],
                "source_examples": a_examples,
                "target_examples": b_examples,
                "source_count": len(a_examples),
                "target_count": len(b_examples),
            })

    print(f"Built {len(experiments)} transfer experiments "
          f"({sum(1 for e in experiments if e['pair']['pair_type'] == 'sibling')} sibling, "
          f"{sum(1 for e in experiments if e['pair']['pair_type'] == 'distant')} distant)")

    return experiments


def evaluate_transfer(
    method_name: str,
    experiment: Dict,
    system,
    data_processor: DataProcessor,
    config: Dict,
    save_path: str,
) -> Dict[str, Any]:
    """
    Run a single transfer experiment.

    Protocol:
    1. Baseline: Evaluate on target without any adaptation
    2. Adapt: Train on source examples
    3. Transfer: Evaluate on target with adapted knowledge

    Args:
        method_name: "ace" or "gsam"
        experiment: Transfer experiment config.
        system: ACE or GSAM system instance.
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
    print(f"  Source examples: {len(source_examples)}")
    print(f"  Target examples: {len(target_examples)}")

    # Step 1: Baseline evaluation on target (no adaptation)
    # This requires a fresh system or resetting the system
    baseline_correct = 0
    for example in target_examples:
        target = example.get("target", "")
        # Simple generation without adaptation
        from utils import extract_answer
        if hasattr(system, 'generator'):
            if hasattr(system, 'knowledge_graph'):
                # GSAM
                ctx, _ = system.graph_retriever.retrieve(
                    example.get("question", ""),
                    example.get("context", ""),
                )
                resp, _, _ = system.generator.generate(
                    question=example.get("question", ""),
                    graph_context=ctx,
                    context=example.get("context", ""),
                )
            else:
                # ACE
                resp, _, _ = system.generator.generate(
                    question=example.get("question", ""),
                    playbook=getattr(system, 'playbook', ''),
                    context=example.get("context", ""),
                )
            answer = extract_answer(resp)
            if data_processor.answer_is_correct(answer, target):
                baseline_correct += 1

    baseline_accuracy = baseline_correct / len(target_examples) if target_examples else 0

    # Step 2: Adapt on source examples
    config_params = {
        'max_num_rounds': config.get('max_num_rounds', 3),
        'curator_frequency': config.get('curator_frequency', 1),
        'token_budget': config.get('playbook_token_budget', 80000),
        'use_json_mode': config.get('json_mode', False),
        'no_ground_truth': config.get('no_ground_truth', False),
    }

    for i, example in enumerate(source_examples):
        print(f"  Adapting on source {i+1}/{len(source_examples)}")
        try:
            if hasattr(system, '_train_single_sample'):
                os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
                system._train_single_sample(
                    task_dict=example,
                    data_processor=data_processor,
                    step_id=f"transfer_{source_concept}_s_{i}",
                    step=i + 1,
                    log_dir=os.path.join(save_path, "logs"),
                    config_params=config_params,
                    total_samples=len(source_examples),
                )
        except Exception as e:
            print(f"  Warning: Adaptation failed on source example {i}: {e}")

    # Step 3: Evaluate on target after adaptation
    transfer_correct = 0
    for example in target_examples:
        target = example.get("target", "")
        try:
            if hasattr(system, 'knowledge_graph'):
                ctx, _ = system.graph_retriever.retrieve(
                    example.get("question", ""),
                    example.get("context", ""),
                )
                resp, _, _ = system.generator.generate(
                    question=example.get("question", ""),
                    graph_context=ctx,
                    context=example.get("context", ""),
                )
            else:
                resp, _, _ = system.generator.generate(
                    question=example.get("question", ""),
                    playbook=getattr(system, 'playbook', ''),
                    context=example.get("context", ""),
                )
            answer = extract_answer(resp)
            if data_processor.answer_is_correct(answer, target):
                transfer_correct += 1
        except Exception as e:
            print(f"  Warning: Evaluation failed: {e}")

    transfer_accuracy = transfer_correct / len(target_examples) if target_examples else 0

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
        "transfer_accuracy": transfer_accuracy,
        **metrics,
    }

    print(f"  Baseline: {baseline_accuracy:.3f} -> Transfer: {transfer_accuracy:.3f}"
          f" (delta={metrics['delta_transfer']:.3f})")

    return result


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


def save_concept_pairs(pairs: List[Dict], output_path: str) -> None:
    """Save concept pairs to JSON."""
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} concept pairs to {output_path}")


def main():
    """Build the FiNER-Transfer benchmark data."""
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
        })

    with open(os.path.join(args.output_dir, "transfer_experiments.json"), "w") as f:
        json.dump(exp_config, f, indent=2)
    print(f"Saved {len(exp_config)} transfer experiments")


if __name__ == "__main__":
    main()
