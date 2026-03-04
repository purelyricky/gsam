#!/usr/bin/env python3
"""
analyze_results.py — View all GSAM experiment results and metrics.

Run after completing experiments to see a summary of all tables and metrics
needed for the paper. Reads from the results/ directory produced by run_gsam.py.

Usage:
    python analyze_results.py                     # look in ./results
    python analyze_results.py --results_dir PATH  # custom results directory
    python analyze_results.py --transfer_dir PATH # custom transfer results dir
"""

import os
import json
import glob
import argparse
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_run(base_dir: str, prefix: str) -> Optional[str]:
    """Return the most recently modified run folder matching prefix."""
    pattern = os.path.join(base_dir, prefix + "*")
    runs = glob.glob(pattern)
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)


def find_results_file(run_dir: str) -> Optional[str]:
    """Find final_results.json inside a run directory (may be nested one level)."""
    # Direct
    p = os.path.join(run_dir, "final_results.json")
    if os.path.exists(p):
        return p
    # One level deep (timestamp subfolder)
    nested = glob.glob(os.path.join(run_dir, "*/final_results.json"))
    if nested:
        return max(nested, key=os.path.getmtime)
    return None


def load_json(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_accuracy(results: Dict, mode: str = "online") -> Optional[float]:
    """Extract accuracy from a final_results.json dict regardless of nesting."""
    if results is None:
        return None
    # ACE stores accuracy at top level
    if "accuracy" in results:
        return results["accuracy"]
    # GSAM stores it nested
    if mode == "online" and "online_test_results" in results:
        return results["online_test_results"].get("accuracy")
    if mode == "offline" and "final_test_results" in results:
        return results["final_test_results"].get("accuracy")
    # Fallback: try any nested key
    for key in ("online_test_results", "final_test_results", "test_results"):
        if key in results:
            acc = results[key].get("accuracy")
            if acc is not None:
                return acc
    return None


def fmt(value: Optional[float], as_pct: bool = False) -> str:
    if value is None:
        return "  N/A  "
    if as_pct:
        return f"{value:.1%}"
    return f"{value:.3f}"


def load_aggregate_metrics(run_dir: str) -> Dict:
    """Load all tracking files from a GSAM run and compute aggregate metrics."""
    from gsam.metrics import aggregate_experiment_results
    # Find the deepest run subfolder that has retrieval_logs or graph_stats
    candidates = [run_dir]
    candidates += glob.glob(os.path.join(run_dir, "*/"))
    for c in sorted(candidates, key=os.path.getmtime, reverse=True):
        if os.path.exists(os.path.join(c, "graph_stats.json")):
            return aggregate_experiment_results(c)
    return {}


# ---------------------------------------------------------------------------
# Main display
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze GSAM experiment results")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Root results directory")
    parser.add_argument("--transfer_dir", type=str, default=None,
                        help="Transfer results directory (default: results_dir/transfer)")
    args = parser.parse_args()

    base = args.results_dir
    transfer_dir = args.transfer_dir or os.path.join(base, "transfer")

    sep = "=" * 65

    # ------------------------------------------------------------------
    # TABLE 1 — Main accuracy results (FiNER and Formula)
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("TABLE 1 — Main Results (Accuracy)")
    print(f"{sep}")
    print(f"  {'System':<30}  {'FiNER Online':>14}  {'Formula Online':>14}")
    print(f"  {'-'*30}  {'-'*14}  {'-'*14}")

    rows = [
        ("ACE (baseline)",       "ace_finer_online",    "ace_formula_online",    "online"),
        ("GSAM (full)",          "gsam_finer_online",   "gsam_formula_online",   "online"),
    ]

    for label, finer_prefix, formula_prefix, mode in rows:
        finer_run  = find_latest_run(base, finer_prefix)
        formula_run = find_latest_run(base, formula_prefix)

        finer_file   = find_results_file(finer_run)  if finer_run   else None
        formula_file = find_results_file(formula_run) if formula_run else None

        finer_acc   = get_accuracy(load_json(finer_file),   mode)
        formula_acc = get_accuracy(load_json(formula_file), mode)

        print(f"  {label:<30}  {fmt(finer_acc):>14}  {fmt(formula_acc):>14}")

    print()
    print("  Source files:  results/*/final_results.json")
    print("  Key (GSAM):    result['online_test_results']['accuracy']")
    print("  Key (ACE):     result['accuracy']")

    # ------------------------------------------------------------------
    # TABLE 2 — GSAM Offline vs. Online (FiNER)
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("TABLE 2 — GSAM Online vs. Offline (FiNER)")
    print(f"{sep}")
    print(f"  {'Run':<30}  {'Accuracy':>10}")
    print(f"  {'-'*30}  {'-'*10}")

    for label, prefix, mode in [
        ("GSAM FiNER Online",  "gsam_finer_online",  "online"),
        ("GSAM FiNER Offline", "gsam_finer_offline", "offline"),
    ]:
        run  = find_latest_run(base, label.lower().replace(" ", "_"))
        if run is None:
            run = find_latest_run(base, prefix)
        f    = find_results_file(run) if run else None
        acc  = get_accuracy(load_json(f), mode)
        print(f"  {label:<30}  {fmt(acc):>10}")

    # ------------------------------------------------------------------
    # TABLE 3 — Ablation Study (FiNER Online)
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("TABLE 3 — Ablation Study (FiNER Online, accuracy)")
    print(f"{sep}")
    print(f"  {'Variant':<35}  {'Accuracy':>10}  {'RFR':>8}  {'Ret.Prec':>10}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*8}  {'-'*10}")

    ablations = [
        ("Full GSAM",                 "gsam_finer_online"),
        ("w/o Ontology",              "ablation_no_ontology"),
        ("w/o Failure Cascades",      "ablation_no_cascades"),
        ("w/o Graph (Embedding only)","ablation_embedding_only"),
        ("w/o Typed Edges",           "ablation_untyped_edges"),
        ("w/o Multi-Epoch Refinement","ablation_no_multiepoch"),
    ]

    for label, prefix in ablations:
        run = find_latest_run(base, prefix)
        f   = find_results_file(run) if run else None
        acc = get_accuracy(load_json(f), "online")

        rfr_val  = None
        prec_val = None
        if run:
            metrics = load_aggregate_metrics(run)
            rfr_data  = metrics.get("rfr_metrics", {})
            prec_data = metrics.get("retrieval_metrics", {})
            rfr_val  = rfr_data.get("rfr")
            prec_val = prec_data.get("mean_precision")

        print(f"  {label:<35}  {fmt(acc):>10}  {fmt(rfr_val):>8}  {fmt(prec_val):>10}")

    print()
    print("  What each metric measures:")
    print("    Accuracy   — Task-level accuracy on FiNER test set")
    print("    RFR        — Repeated Failure Rate: fraction of errors that repeat a")
    print("                 previously-seen failure pattern. Lower is better.")
    print("    Ret.Prec   — Retrieval Precision: fraction of retrieved graph nodes")
    print("                 actually referenced in the generator's reasoning. Higher = more focused.")

    # ------------------------------------------------------------------
    # TABLE 4 — Graph Growth
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("TABLE 4 — Knowledge Graph Statistics (Full GSAM FiNER Online)")
    print(f"{sep}")

    gsam_run = find_latest_run(base, "gsam_finer_online")
    if gsam_run:
        metrics = load_aggregate_metrics(gsam_run)
        gs = metrics.get("graph_stats", {})
        if gs:
            print(f"  Total nodes:       {gs.get('total_nodes', 'N/A')}")
            print(f"  Total edges:       {gs.get('total_edges', 'N/A')}")
            nc = gs.get("node_counts", {})
            print(f"  Strategy nodes:    {nc.get('Strategy', 0)}")
            print(f"  AntiPattern nodes: {nc.get('AntiPattern', 0)}")
            print(f"  Confusion nodes:   {nc.get('Confusion', 0)}")
            print(f"  Concept nodes:     {nc.get('Concept', 0)}")
            print(f"  Formula nodes:     {nc.get('Formula', 0)}")
            cov = gs.get("concept_coverage", None)
            print(f"  Concept coverage:  {fmt(cov, as_pct=True)}")
            print(f"    (fraction of 139 XBRL entity types with >= 1 Strategy node)")
        else:
            print("  No graph_stats.json found in run directory.")
    else:
        print("  No GSAM FiNER online run found in results/")

    # ------------------------------------------------------------------
    # TABLE 5 — Transfer Metrics (FiNER-Transfer)
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("TABLE 5 — FiNER-Transfer Benchmark (H1: Cross-Concept Transfer)")
    print(f"{sep}")

    agg_path = os.path.join(transfer_dir, "aggregate_metrics.json")
    agg = load_json(agg_path)
    if agg:
        print(f"  Near-transfer rate (sibling pairs):  {fmt(agg.get('near_transfer_rate'), as_pct=True)}")
        print(f"  Far-transfer rate (distant pairs):   {fmt(agg.get('far_transfer_rate'), as_pct=True)}")
        print(f"  Negative transfer rate:              {fmt(agg.get('negative_transfer_rate'), as_pct=True)}")
        print(f"  Transfer precision (mean delta):     {fmt(agg.get('transfer_precision'))}")
        print(f"  Total pairs evaluated:               {agg.get('total_pairs', 'N/A')}")
        print(f"    Sibling pairs:                     {agg.get('sibling_pairs', 'N/A')}")
        print(f"    Distant pairs:                     {agg.get('distant_pairs', 'N/A')}")
        print()
        print("  What these measure (H1 hypothesis):")
        print("    Near-transfer rate  — Fraction of sibling concept pairs where adapting on A")
        print("                          improved performance on B. GSAM > ACE proves H1.")
        print("    Far-transfer rate   — Same for distant (cross-category) pairs.")
        print("                          Expected to be low; serves as negative control.")
        print("    Negative transfer   — Fraction of pairs where adaptation hurt performance.")
        print("    Transfer precision  — Mean accuracy delta for pairs with positive transfer.")
    else:
        print(f"  No aggregate_metrics.json found at: {agg_path}")
        print("  Run Step 5 from the README to generate transfer results.")

    # ------------------------------------------------------------------
    # TABLE 6 — Latency Breakdown (seconds/sample)
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("TABLE 6 — Latency Breakdown (seconds/sample)")
    print(f"{sep}")
    print(f"  {'Method':<30}  {'Generator':>10}  {'Reflector':>10}  {'Curator':>10}  "
          f"{'Retrieval':>10}  {'Graph Upd':>10}  {'Total':>8}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    latency_rows = [
        ("ACE (baseline)",  "ace_finer_online",  "online"),
        ("GSAM (full)",     "gsam_finer_online", "online"),
    ]

    def get_latency(results_dict):
        """Extract latency_stats from a final_results.json dict."""
        if results_dict is None:
            return None
        # Top-level (written by run() for both ACE and GSAM)
        if "latency_stats" in results_dict:
            return results_dict["latency_stats"]
        # Fallback: nested inside online_test_results (GSAM legacy)
        for key in ("online_test_results", "final_test_results", "training_results"):
            sub = results_dict.get(key, {})
            if isinstance(sub, dict) and "latency_stats" in sub:
                return sub["latency_stats"]
        return None

    def fmt_lat(v):
        if v is None:
            return "   N/A   "
        return f"{v:.3f}"

    for label, prefix, mode in latency_rows:
        run = find_latest_run(base, prefix)
        f = find_results_file(run) if run else None
        data = load_json(f)
        lat = get_latency(data)

        if lat:
            gen  = lat.get("generator_mean_s")
            ref  = lat.get("reflector_mean_s")
            cur  = lat.get("curator_mean_s")
            ret  = lat.get("retrieval_mean_s")
            gupd = lat.get("graph_update_mean_s")
            tot  = lat.get("total_per_sample_mean_s")
        else:
            gen = ref = cur = ret = gupd = tot = None

        print(f"  {label:<30}  {fmt_lat(gen):>10}  {fmt_lat(ref):>10}  {fmt_lat(cur):>10}  "
              f"{fmt_lat(ret):>10}  {fmt_lat(gupd):>10}  {fmt_lat(tot):>8}")

    print()
    print("  All times in seconds per training sample (mean over run).")
    print("  Retrieval and Graph Update are 0.0 for ACE (no graph component).")
    print("  Source key: result['latency_stats'] in final_results.json")

    # ------------------------------------------------------------------
    # SUMMARY — Key metrics for paper hypotheses
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("PAPER HYPOTHESES — Key Evidence Locations")
    print(f"{sep}")
    print()
    print("  H1: Graph-structured memory improves cross-concept transfer")
    print("      → Table 5 (near_transfer_rate GSAM > ACE)")
    print("      → Table 1 (GSAM accuracy > ACE accuracy on FiNER)")
    print()
    print("  H2: Failure cascades reduce error repetition")
    print("      → Table 3 (RFR: Full GSAM < w/o Failure Cascades)")
    print("      → Compare: ablation_no_cascades vs gsam_finer_online")
    print()
    print("  H3: Ontology-aware retrieval improves retrieval precision")
    print("      → Table 3 (Ret.Prec: Full GSAM > Embedding only)")
    print("      → Compare: ablation_embedding_only vs gsam_finer_online")
    print()
    print("  Graph quality evidence:")
    print("      → Table 4 (concept_coverage after training)")
    print("      → graph_checkpoints/ for evolution over training steps")
    print()
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
