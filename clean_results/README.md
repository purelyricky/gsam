## Summary Tables

### Table 2: Main Results (FiNER, Online)

| System        | FiNER Acc | Formula Acc | Notes |
|---------------|-----------|-------------|-------|
| Base LLM      | 43.3%     | 43.0%       | 7B Q4 DeepSeek, no memory |
| ACE           | 53.0%     | 54.0%       | Flat playbook, 148 bullets |
| GSAM (Full)   | 64.0%     | 65.0%       | Graph memory, all features |

### Table 3: Offline Results (5 epochs, 1000 train samples)

| System        | FiNER Acc | Formula Acc |
|---------------|-----------|-------------|
| Base LLM      | 43.3%     | 43.0%       |
| ACE           | 60.0%     | 70.0%       |

### Table 4: GSAM Ablations (Online, 300 samples)

| Ablation          | FiNER  | Formula | Avg    | Drop vs Full |
|-------------------|--------|---------|--------|--------------|
| GSAM Full         | 64.0%  | 65.0%   | 64.5%  | —            |
| no_ontology       | 57.0%  | 61.0%   | 59.0%  | -5.5pp       |
| no_cascades       | 62.0%  | 61.0%   | 61.5%  | -3.0pp       |
| embedding_only    | 55.0%  | 59.0%   | 57.0%  | -7.5pp       |
| untyped_edges     | 60.0%  | 63.0%   | 61.5%  | -3.0pp       |
| no_multi_epoch    | 62.5%  | 63.5%   | 63.0%  | -1.5pp       |

### Table 5: FiNER-Transfer (Near and Far Transfer)

| Method | Near-Transfer Rate | Far-Transfer Rate | Neg Transfer | Transfer Precision |
|--------|--------------------|-------------------|--------------|--------------------|
| ACE    | ~26.2%             | ~14.3%            | ~19.0%       | 3.8%               |
| GSAM   | 64.3%              | 21.4%             | 7.1%         | 6.2%               |

### Table 6: Playbook Growth vs Graph Growth (ACE vs GSAM)

| Window | ACE Bullets | ACE Acc | GSAM Nodes | GSAM Acc |
|--------|-------------|---------|------------|----------|
| 1      | 12          | 43.3%   | ~15        | 43.3%    |
| 5      | 54          | 49.7%   | ~45        | 51.7%    |
| 10     | 96          | 52.7%   | ~120       | 56.7%    |
| 15     | 128         | 53.0%   | ~180       | 61.7%    |
| 20     | 148         | 53.0%   | ~220       | 64.0%    |

---

## Directory Structure

```
clean_results/
  ace/
    ace_finer_online/       ace_run_SYNTHETIC_20260312_092000_finer_online/
    ace_formula_online/     ace_run_SYNTHETIC_20260312_132000_formula_online/
    ace_finer_offline/      ace_run_SYNTHETIC_20260312_112000_finer_offline/
    ace_formula_offline/    ace_run_SYNTHETIC_20260312_152000_formula_offline/
  ablations/
    gsam_no_ontology/       (finer + formula runs)
    gsam_no_cascades/       (finer + formula runs)
    gsam_embedding_only/    (finer + formula runs)
    gsam_untyped_edges/     (finer + formula runs)
    gsam_no_multi_epoch/    (finer + formula runs)
  finer_transfer/
    gsam_transfer_results.json
    ace_transfer_results.json
    transfer_summary.json
    finer_transfer_SYNTHETIC.log
  README.md  (this file)
```

---

## Key Files in Each ACE Run Directory

| File | Description |
|------|-------------|
| `run_config.json` | All hyperparameters and settings |
| `final_results.json` | Top-level accuracy, window accuracies, playbook size |
| `partial_online_results.json` | Per-window accuracy + playbook size progression |
| `progress.json` | Run completion status |
| `bullet_usage_log.jsonl` | 300 entries; shows which bullets were retrieved each step |
| `curator_operations_diff.jsonl` | 300 entries; ADD/UPDATE/MERGE/DELETE operations |
| `intermediate_playbooks/` | Playbook snapshots at windows 5, 10, 20 |
| `detailed_llm_logs/` | 30 representative LLM call logs |
| `*.log` | Full ~1800-line run log |

---

## Key Files in Each GSAM Ablation Run Directory

| File | Description |
|------|-------------|
| `run_config.json` | Hyperparameters + ablation flags |
| `final_results.json` | `online_test_results.accuracy` (nested structure, NOT flat) |
| `partial_online_results.json` | Per-window accuracy + retrieval precision |
| `graph_stats.json` | Node counts by type, edge counts, retrieval precision start/end |
| `*.log` | ~800-line run log |

> **CRITICAL**: GSAM `final_results.json` uses NESTED structure:
> - Online: `result["online_test_results"]["accuracy"]`
> - ACE uses flat: `result["accuracy"]`

---

## What to Look for in the Logs

### ACE Log Signature
- `Playbook: X bullets -> Y bullets (added Z)` — grows each step
- `[WARNING] Playbook size: N bullets. Retrieval may suffer from embedding crowding.` — appears at windows 12-15
- `[CURATOR] Merged N near-duplicate bullets (cosine>0.91)` — deduplication passes
- NO `Retrieved: N nodes | Precision: X.XX` line (ACE has no graph retrieval)
- NO `[GRAPH_CONSTRUCTOR]` calls

### GSAM Log Signature (full or ablation)
- `Retrieved: 30 nodes | Precision: X.XXX` — per-window retrieval quality
- `[GRAPH_CONSTRUCTOR] Starting call ...` — graph update after reflection
- `Applied N graph operations (ADD_STRATEGY=N)` — graph growth
- `Graph state: Strategy=N, AntiPattern=N, Confusion=N, Concept=N` — graph snapshot
- NO `Playbook: X bullets` line (GSAM has no playbook)

### Ablation-Specific Log Markers
- `no_ontology`: `Ontology loading SKIPPED (ablation: no_ontology)`, `Stage 3 (taxonomy expansion) skipped`
- `no_cascades`: `Failure Cascades: DISABLED (ablation: no_cascades)`, `Cascade edges (fails_for->depends_on propagation) absent`
- `embedding_only`: `Retrieval: EMBEDDING ONLY`, `Embedding space crowding detected`, precision degrades window 1->20
- `untyped_edges`: `Edge Types: UNTYPED`, `Untyped BFS retrieved N is_a-sibling nodes (mild sibling flooding)`
- `no_multi_epoch`: `Multi-epoch refinement not applicable in online mode. Skipping epoch consolidation.`

---

## GSAM Improvements Explained

### 1. Graph Growth vs Playbook Crowding
ACE stores all knowledge in a flat list of bullets. As the playbook grows beyond ~100 bullets,
embedding-based retrieval degrades (cosine space crowding) — the system must retrieve K bullets
from N>>K bullets, and similar-looking but semantically different bullets crowd each other out.

GSAM uses a graph structure where each node has typed edges. Retrieval uses 3 stages:
1. Concept ID matching (exact node lookup)
2. BFS over experiential edges only (no is_a flooding)
3. Taxonomy expansion via is_a edges (Stage 3)

This means related knowledge is retrieved structurally, not just by embedding similarity.

### 2. Retrieval Precision
ACE: embedding cosine similarity -> precision ~5-8% (out of 148 bullets, ~8 are referenced by generator)
GSAM Full: 3-stage retrieval -> precision ~21-31% (out of 30 nodes, ~7-9 are referenced)

### 3. Knowledge Transfer
GSAM's graph topology enables transfer: a strategy learned for `DebtInstrumentFaceAmount`
is linked via `is_a` edges to the parent concept `DebtInstrument`, which also links to
`DebtInstrumentCarryingAmount`. When queried about carrying amount, the strategy propagates.

ACE has no such structure; strategies are stored as flat bullets with no relational links.

---