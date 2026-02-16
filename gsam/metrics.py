"""
GSAM Metrics Module

Implements GSAM-specific evaluation metrics:
- Repeated Failure Rate (RFR)
- Retrieval Precision
- Concept Coverage
- Graph Quality Metrics
- Transfer Metrics
"""

import json
from typing import Dict, List, Any, Optional
from collections import defaultdict


def compute_repeated_failure_rate(error_history: List[Dict]) -> Dict[str, Any]:
    """
    Compute Repeated Failure Rate (RFR).

    RFR = |{t_j : error(t_j) = error(t_i), i < j}| /
          |{t_j : could_have_used(error(t_i)), i < j}|

    An error is considered "repeated" if the same concept(s) are involved
    in a failure that was seen in an earlier task.

    Args:
        error_history: List of error records with:
            - step: int
            - is_correct: bool
            - concepts_involved: List[str]
            - confusion_pairs: List[List[str]]

    Returns:
        Dict with RFR metrics.
    """
    if not error_history:
        return {
            "rfr": 0.0,
            "total_errors": 0,
            "repeated_errors": 0,
            "unique_error_patterns": 0,
            "opportunities": 0,
        }

    # Track seen error patterns
    seen_concept_errors = set()  # frozenset of concepts
    seen_confusion_pairs = set()  # frozenset of pairs

    repeated_count = 0
    total_errors = 0
    opportunities = 0

    for entry in error_history:
        if entry.get("is_correct", True):
            continue

        total_errors += 1
        concepts = tuple(sorted(entry.get("concepts_involved", [])))
        confusions = tuple(
            tuple(sorted(pair))
            for pair in entry.get("confusion_pairs", [])
        )

        # Check if this error pattern was seen before
        concept_key = frozenset(concepts) if concepts else None
        is_repeated = False

        if concept_key and concept_key in seen_concept_errors:
            is_repeated = True

        for conf_pair in confusions:
            if frozenset(conf_pair) in seen_confusion_pairs:
                is_repeated = True

        if is_repeated:
            repeated_count += 1

        # Count opportunity (could have been warned about this error)
        if concept_key and concept_key in seen_concept_errors:
            opportunities += 1
        elif concept_key:
            opportunities += 1  # First occurrence is also an opportunity

        # Record this error pattern
        if concept_key:
            seen_concept_errors.add(concept_key)
        for conf_pair in confusions:
            seen_confusion_pairs.add(frozenset(conf_pair))

    rfr = repeated_count / opportunities if opportunities > 0 else 0.0

    return {
        "rfr": rfr,
        "total_errors": total_errors,
        "repeated_errors": repeated_count,
        "unique_error_patterns": len(seen_concept_errors),
        "opportunities": opportunities,
    }


def compute_retrieval_precision(retrieval_logs: List[Dict]) -> Dict[str, Any]:
    """
    Compute retrieval precision from retrieval logs.

    Precision = |referenced nodes| / |retrieved nodes|

    Args:
        retrieval_logs: List of per-task retrieval records with:
            - retrieved_count: int
            - referenced_count: int
            - precision: float

    Returns:
        Dict with precision metrics.
    """
    if not retrieval_logs:
        return {"mean_precision": 0.0, "total_tasks": 0}

    precisions = [entry.get("precision", 0.0) for entry in retrieval_logs]
    retrieval_times = [entry.get("retrieval_time_s", 0.0) for entry in retrieval_logs]

    return {
        "mean_precision": sum(precisions) / len(precisions),
        "total_tasks": len(retrieval_logs),
        "precisions_by_quartile": {
            "q1": sorted(precisions)[len(precisions) // 4],
            "q2": sorted(precisions)[len(precisions) // 2],
            "q3": sorted(precisions)[3 * len(precisions) // 4],
        },
        "mean_retrieval_time_s": sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0,
    }


def compute_concept_coverage(graph_stats: Dict) -> float:
    """
    Compute concept coverage: fraction of XBRL entity types with >= 1 strategy.

    Args:
        graph_stats: Output of KnowledgeGraph.stats().

    Returns:
        Coverage fraction (0.0 to 1.0).
    """
    return graph_stats.get("concept_coverage", 0.0)


def compute_transfer_metrics(
    source_accuracy: float,
    target_accuracy_with_transfer: float,
    target_accuracy_without_transfer: float,
) -> Dict[str, float]:
    """
    Compute cross-concept transfer metrics for FiNER-Transfer.

    Args:
        source_accuracy: Accuracy on source concept after adaptation.
        target_accuracy_with_transfer: Accuracy on target with transferred knowledge.
        target_accuracy_without_transfer: Accuracy on target without any adaptation.

    Returns:
        Dict with transfer metrics.
    """
    delta_transfer = target_accuracy_with_transfer - target_accuracy_without_transfer
    return {
        "source_accuracy": source_accuracy,
        "target_with_transfer": target_accuracy_with_transfer,
        "target_without_transfer": target_accuracy_without_transfer,
        "delta_transfer": delta_transfer,
        "positive_transfer": delta_transfer > 0,
        "negative_transfer": delta_transfer < 0,
    }


def aggregate_experiment_results(
    results_dir: str,
) -> Dict[str, Any]:
    """
    Aggregate results from a GSAM experiment run.

    Loads all tracking logs and computes summary metrics.

    Args:
        results_dir: Path to the experiment results directory.

    Returns:
        Dict with aggregated metrics.
    """
    import os

    summary = {}

    # Load retrieval logs
    retrieval_path = os.path.join(results_dir, "retrieval_logs.jsonl")
    if os.path.exists(retrieval_path):
        retrieval_logs = []
        with open(retrieval_path) as f:
            for line in f:
                if line.strip():
                    retrieval_logs.append(json.loads(line))
        summary["retrieval_metrics"] = compute_retrieval_precision(retrieval_logs)

    # Load error tracking
    error_path = os.path.join(results_dir, "error_tracking.jsonl")
    if os.path.exists(error_path):
        error_history = []
        with open(error_path) as f:
            for line in f:
                if line.strip():
                    error_history.append(json.loads(line))
        summary["rfr_metrics"] = compute_repeated_failure_rate(error_history)

    # Load graph stats
    graph_stats_path = os.path.join(results_dir, "graph_stats.json")
    if os.path.exists(graph_stats_path):
        with open(graph_stats_path) as f:
            graph_stats = json.load(f)
        summary["graph_stats"] = graph_stats
        summary["concept_coverage"] = compute_concept_coverage(graph_stats)

    # Load final results
    final_path = os.path.join(results_dir, "final_results.json")
    if os.path.exists(final_path):
        with open(final_path) as f:
            summary["final_results"] = json.load(f)

    return summary
