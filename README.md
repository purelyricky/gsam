# GSAM: Graph-Structured Adaptive Memory for Agentic Context Engineering

<div align="left">

<img src="assets/images/ace_framework.png" alt="ACE Framework" width="800"/>

</div>

---

## Overview

**GSAM (Graph-Structured Adaptive Memory)** extends the [ACE (Agentic Context Engineering)](https://arxiv.org/abs/2510.04618) framework by replacing its flat bullet-point playbook with an **ontology-grounded knowledge graph**. Where ACE stores strategies as an unstructured list of text bullets, GSAM organizes knowledge into **typed nodes** (Strategy, AntiPattern, Concept, Formula, Confusion) connected by **typed edges** (is_a, applies_to, fails_for, fixes, confused_with, etc.), enabling:

- **Ontology-aware retrieval**: Query-time subgraph extraction via concept matching + BFS traversal, so only structurally relevant strategies are surfaced
- **Failure cascade warnings**: When an anti-pattern targets a concept that a formula depends on, GSAM warns the generator that the formula result will also be wrong
- **Cross-concept transfer**: Strategies learned for one XBRL entity type (e.g., `Revenues`) are automatically proposed as tentative candidates for its taxonomy siblings (e.g., `RevenueFromContractWithCustomerExcludingAssessedTax`)
- **Concept confusion tracking**: Explicit Confusion nodes document commonly confused entity pairs with distinguishing criteria

GSAM is evaluated on the **FiNER** (financial named entity recognition in XBRL) and **XBRL Formula** tasks from the ACE paper, with a new **FiNER-Transfer** benchmark that tests cross-concept transfer.

## Repository Structure

```
gsam/
├── gsam/                              # GSAM extension module
│   ├── __init__.py
│   ├── graph_memory.py                # KnowledgeGraph: typed nodes/edges via NetworkX
│   ├── gsam.py                        # GSAM orchestrator (extends ACE architecture)
│   ├── ontology.py                    # XBRL taxonomy parser -> Concept nodes + is_a edges
│   ├── metrics.py                     # RFR, retrieval precision, concept coverage, transfer
│   ├── core/
│   │   ├── graph_constructor.py       # Curator deltas -> typed graph operations
│   │   └── graph_retriever.py         # 3-stage ontology-aware subgraph retrieval
│   └── prompts/
│       ├── generator.py               # Generator prompt consuming serialized subgraphs
│       ├── reflector.py               # Reflector prompt with concept-level error analysis
│       ├── curator.py                 # Curator prompt outputting graph operations
│       └── graph_constructor.py       # Fallback entity/relationship extraction prompt
│
├── ace/                               # Original ACE framework (baseline)
│   ├── ace.py                         # ACE orchestrator
│   ├── core/                          # Generator, Reflector, Curator agents
│   └── prompts/                       # ACE prompt templates
│
├── eval/finance/                      # Financial domain evaluation
│   ├── data_processor.py              # FiNER + Formula data preprocessing and scoring
│   ├── run.py                         # ACE entry point (baseline)
│   ├── run_gsam.py                    # GSAM entry point
│   ├── finer_transfer.py              # FiNER-Transfer benchmark builder
│   └── data/
│       ├── sample_config.json         # Data paths for finer and formula tasks
│       ├── xbrl_taxonomy.json         # US-GAAP taxonomy (139 entities, 9 categories)
│       ├── finer_train_batched_1000_samples.jsonl
│       ├── finer_val_batched_500_samples.jsonl
│       ├── finer_test_subset_006_seed42.jsonl
│       ├── formula_train_subset_500.jsonl
│       ├── formula_val_subset_300.jsonl
│       ├── formula_test.jsonl
│       └── finer_transfer/            # Generated transfer experiment splits
│
├── experiments/
│   ├── run_experiment.py              # Experiment runner (single or suite)
│   └── configs/                       # JSON experiment configs
│       ├── gsam_finer_offline.json
│       ├── gsam_finer_online.json
│       ├── gsam_ablation_no_ontology.json
│       ├── gsam_ablation_no_cascades.json
│       ├── gsam_ablation_embedding_only.json
│       └── gsam_ablation_untyped_edges.json
│
├── tests/                             # Unit tests (57 tests)
│   ├── test_graph_memory.py           # KnowledgeGraph CRUD, retrieval, serialization
│   ├── test_metrics.py                # RFR, precision, transfer metrics
│   └── test_ontology.py              # Taxonomy loading, is_a edges, sibling resolution
│
├── llm.py                            # LLM call utilities
├── logger.py                         # Logging utilities
├── utils.py                          # Shared utilities (evaluate_test_set, extract_answer)
├── playbook_utils.py                 # ACE playbook operations (used by GraphConstructor)
├── requirements.txt                  # Python dependencies
└── EXTENDING_ACE.md                  # Guide for adding new tasks to ACE
```

## Installation

### 1. Clone and install dependencies

```bash
git clone https://github.com/purelyricky/gsam.git
cd gsam
pip install -r requirements.txt
```

The key dependencies are:

| Package | Purpose |
|---------|---------|
| `networkx>=3.0` | Knowledge graph storage and traversal |
| `numpy>=1.24.0` | Embedding computations |
| `sentence-transformers>=2.2.0` | Node embedding similarity for deduplication and retrieval |
| `openai>=1.0.0` | LLM API client (OpenAI-compatible) |
| `tiktoken` | Token counting for budget management |

### 2. Configure API keys

Create a `.env` file in the project root with your API key for one of the supported providers:

```bash
# Option A: SambaNova (default)
SAMBANOVA_API_KEY=your_key_here

# Option B: Together AI
TOGETHER_API_KEY=your_key_here

# Option C: OpenAI
OPENAI_API_KEY=your_key_here
```

### 3. Verify installation

```bash
# Run the unit tests to confirm everything is working
python -m unittest tests.test_graph_memory tests.test_metrics tests.test_ontology -v
```

You should see all 57 tests pass.

## How GSAM Works

GSAM follows the same Generator-Reflector-Curator loop as ACE, but replaces the flat playbook with a knowledge graph:

### Training Loop (per sample)

```
1. RETRIEVE    Query text -> concept matching + BFS -> serialized subgraph
2. GENERATE    Subgraph context + question -> LLM -> predicted answer
3. REFLECT     Prediction vs. ground truth -> concept-level error analysis
4. CURATE      Reflection -> structured graph operations (ADD_STRATEGY, ADD_ANTIPATTERN, ADD_CONFUSION)
5. CONSTRUCT   Operations -> deduplicate -> apply to knowledge graph
6. PRUNE       Periodically remove low-utility nodes (never prune ontology backbone)
```

### Knowledge Graph Node Types

| Type | Prefix | Description |
|------|--------|-------------|
| **Concept** | `C:` | XBRL entity types from the taxonomy (e.g., `Revenues`, `NetIncomeLoss`) |
| **Strategy** | `S:` | Learned approaches that work (e.g., "Check for unrealized gains to distinguish NetIncome from ComprehensiveIncome") |
| **AntiPattern** | `A:` | Known failure modes to avoid (e.g., "Confusing gross revenue with net revenue") |
| **Formula** | `F:` | XBRL formula definitions with their input dependencies |
| **Confusion** | `X:` | Documented concept pairs that are commonly confused |

### Knowledge Graph Edge Types

| Edge | Direction | Meaning |
|------|-----------|---------|
| `is_a` | Entity -> Subcategory -> Category | Taxonomy hierarchy |
| `applies_to` | Strategy -> Concept | Strategy addresses this concept |
| `fails_for` | AntiPattern -> Concept | Anti-pattern causes errors for this concept |
| `fixes` | Strategy -> AntiPattern | Strategy resolves this anti-pattern |
| `depends_on` | Formula -> Concept | Formula requires this entity as input |
| `confused_with` | Concept <-> Concept | Bidirectional confusion link |
| `conflicts_with` | Strategy <-> Strategy | Mutually exclusive strategies |

### Three-Stage Retrieval

```
Stage 1: Concept Identification
  - Keyword matching: exact concept names found in query text
  - Embedding similarity: cosine similarity against all Concept node embeddings

Stage 2: Graph Traversal
  - BFS from matched concepts (default depth=2)
  - Collects connected Strategies, AntiPatterns, Formulas

Stage 3: Taxonomic Expansion
  - Walk is_a edges to find sibling concepts
  - Transfer strategies from siblings (marked as "tentative")
  - Include failure cascade warnings for formulas
```

## Quick Start

### Run GSAM on FiNER (online mode)

This trains and evaluates GSAM in a single pass over the test set, updating the knowledge graph after each window:

```bash
python -m eval.finance.run_gsam \
    --task_name finer \
    --mode online \
    --save_path results/gsam_finer_online \
    --api_provider sambanova \
    --taxonomy_path ./eval/finance/data/xbrl_taxonomy.json
```

### Run GSAM on FiNER (offline mode)

This trains on the training set with validation checkpoints, then evaluates on the test set:

```bash
python -m eval.finance.run_gsam \
    --task_name finer \
    --mode offline \
    --save_path results/gsam_finer_offline \
    --api_provider sambanova \
    --taxonomy_path ./eval/finance/data/xbrl_taxonomy.json
```

### Run GSAM on XBRL Formula

```bash
python -m eval.finance.run_gsam \
    --task_name formula \
    --mode online \
    --save_path results/gsam_formula_online \
    --api_provider sambanova \
    --taxonomy_path ./eval/finance/data/xbrl_taxonomy.json
```

### Run ACE baseline for comparison

```bash
python -m eval.finance.run \
    --task_name finer \
    --mode online \
    --save_path results/ace_finer_online \
    --api_provider sambanova
```

### Evaluate a saved graph (no training)

```bash
python -m eval.finance.run_gsam \
    --task_name finer \
    --mode eval_only \
    --save_path results/eval_results \
    --api_provider sambanova \
    --taxonomy_path ./eval/finance/data/xbrl_taxonomy.json
```

### Smoke test with limited samples

Use `--max_samples` to quickly verify everything works before a full run:

```bash
python -m eval.finance.run_gsam \
    --task_name finer \
    --mode online \
    --save_path results/smoke_test \
    --api_provider sambanova \
    --taxonomy_path ./eval/finance/data/xbrl_taxonomy.json \
    --max_samples 10
```

## Programmatic Usage

```python
from gsam import GSAM
from eval.finance.data_processor import DataProcessor

# Initialize GSAM with ontology
gsam_system = GSAM(
    api_provider="sambanova",
    generator_model="DeepSeek-V3.1",
    reflector_model="DeepSeek-V3.1",
    curator_model="DeepSeek-V3.1",
    max_tokens=4096,
    taxonomy_path="./eval/finance/data/xbrl_taxonomy.json",
    merge_threshold=0.9,       # Cosine similarity for node deduplication
    retrieval_depth=2,         # BFS depth for graph retrieval
    prune_frequency=50,        # Prune low-utility nodes every N steps
)

# Prepare data
processor = DataProcessor(task_name="finer")

# Configure and run
config = {
    'num_epochs': 1,
    'max_num_rounds': 3,
    'curator_frequency': 1,
    'eval_steps': 100,
    'online_eval_frequency': 15,
    'save_steps': 50,
    'playbook_token_budget': 80000,
    'task_name': 'finer',
    'mode': 'online',
    'json_mode': False,
    'no_ground_truth': False,
    'save_dir': './results',
    'test_workers': 20,
}

results = gsam_system.run(
    mode='online',
    test_samples=test_data,
    data_processor=processor,
    config=config,
)

# Inspect the evolved knowledge graph
print(gsam_system.knowledge_graph)
# KnowledgeGraph(nodes=245, edges=412, concepts=139, coverage=0.67)

stats = gsam_system.knowledge_graph.stats()
print(f"Strategies: {stats['node_counts'].get('Strategy', 0)}")
print(f"Anti-patterns: {stats['node_counts'].get('AntiPattern', 0)}")
print(f"Concept coverage: {stats['concept_coverage']:.2%}")

# Save/load the graph
gsam_system.knowledge_graph.save("my_graph.json")

from gsam.graph_memory import KnowledgeGraph
loaded_graph = KnowledgeGraph.load("my_graph.json")
```

## Running Experiments

### Using experiment configs

Each experiment is defined by a JSON config file in `experiments/configs/`. The experiment runner handles argument passing:

```bash
# Run a single experiment
python -m experiments.run_experiment \
    --config experiments/configs/gsam_finer_online.json \
    --save_path results

# Run all ablation experiments
python -m experiments.run_experiment \
    --config_dir experiments/configs/ \
    --save_path results \
    --filter ablation

# Run everything
python -m experiments.run_experiment \
    --config_dir experiments/configs/ \
    --save_path results
```

### Available experiments

| Config | What it tests |
|--------|---------------|
| `gsam_finer_offline.json` | Full GSAM with ontology, offline training on FiNER |
| `gsam_finer_online.json` | Full GSAM with ontology, online adaptation on FiNER |
| `gsam_ablation_no_ontology.json` | Ablation: no XBRL taxonomy initialization (graph starts empty) |
| `gsam_ablation_no_cascades.json` | Ablation: no failure cascade edges or AntiPattern nodes |
| `gsam_ablation_embedding_only.json` | Ablation: retrieval by embedding similarity only (no graph BFS) |
| `gsam_ablation_untyped_edges.json` | Ablation: all edges become generic `related_to` (no typed edges) |

### Ablation study rationale

Each ablation isolates one component of GSAM to measure its contribution:

- **No ontology** (`--no_ontology`): Tests whether pre-loading the XBRL taxonomy as Concept nodes improves accuracy vs. learning concepts from scratch. Expected impact: weaker concept matching, no sibling transfer.

- **No failure cascades** (`--no_failure_cascades`): Tests whether tracking anti-patterns and their cascading effects through formula dependencies reduces repeated errors. Expected impact: higher Repeated Failure Rate (RFR).

- **Embedding-only retrieval** (`--embedding_only_retrieval`): Tests whether graph structure adds value beyond embedding similarity. Retrieves top-k strategies by cosine similarity without BFS traversal or taxonomic expansion. Expected impact: lower retrieval precision, no cross-concept transfer.

- **Untyped edges** (`--untyped_edges`): Tests whether typed edges (is_a, applies_to, fails_for) matter vs. generic connections. Expected impact: noisier retrieval, weaker serialized context.

## FiNER-Transfer Benchmark

The FiNER-Transfer benchmark measures cross-concept transfer: whether adapting on one XBRL entity type improves performance on a related entity type.

### Step 1: Build the benchmark

```bash
python -m eval.finance.finer_transfer \
    --taxonomy_path ./eval/finance/data/xbrl_taxonomy.json \
    --finer_data_path ./eval/finance/data/finer_train_batched_1000_samples.jsonl \
    --output_dir ./eval/finance/data/finer_transfer
```

This generates:
- `concept_pairs.json`: All sibling pairs (same subcategory) and distant pairs (different categories)
- `transfer_experiments.json`: Experiment configs with source/target counts

### Step 2: Run transfer experiments

The transfer evaluation protocol for each concept pair (A, B):

1. **Baseline**: Evaluate on concept B with no adaptation
2. **Adapt**: Train on concept A examples (the source)
3. **Transfer**: Evaluate on concept B again with the adapted graph

```python
from eval.finance.finer_transfer import (
    load_taxonomy, build_concept_pairs, build_transfer_splits,
    evaluate_transfer, compute_aggregate_transfer_metrics,
)
from eval.finance.data_processor import DataProcessor
from gsam import GSAM

# Build experiment pairs
taxonomy = load_taxonomy("./eval/finance/data/xbrl_taxonomy.json")
pairs = build_concept_pairs(taxonomy)

# Initialize system
gsam = GSAM(
    api_provider="sambanova",
    generator_model="DeepSeek-V3.1",
    reflector_model="DeepSeek-V3.1",
    curator_model="DeepSeek-V3.1",
    taxonomy_path="./eval/finance/data/xbrl_taxonomy.json",
)

processor = DataProcessor(task_name="finer")
config = {'max_num_rounds': 3, 'curator_frequency': 1, 'playbook_token_budget': 80000}

# Run transfer experiments
results = []
for experiment in experiments[:5]:  # First 5 pairs
    result = evaluate_transfer("gsam", experiment, gsam, processor, config, "results/transfer")
    results.append(result)

# Aggregate metrics
agg = compute_aggregate_transfer_metrics(results)
print(f"Near-transfer rate: {agg['near_transfer_rate']:.2%}")
print(f"Far-transfer rate: {agg['far_transfer_rate']:.2%}")
print(f"Negative transfer rate: {agg['negative_transfer_rate']:.2%}")
```

## GSAM-Specific Metrics

GSAM introduces several metrics beyond standard accuracy. After a run completes, use the metrics module to compute them:

```python
from gsam.metrics import (
    compute_repeated_failure_rate,
    compute_retrieval_precision,
    compute_concept_coverage,
    aggregate_experiment_results,
)

# Aggregate everything from a results directory
summary = aggregate_experiment_results("results/gsam_finer_online/gsam_run_TIMESTAMP_finer_online/")

# Or compute individually:

# 1. Repeated Failure Rate (RFR)
#    Measures how often the same conceptual error recurs.
#    Lower is better; GSAM should reduce this by anchoring
#    anti-patterns to specific concepts.
rfr = summary["rfr_metrics"]
print(f"RFR: {rfr['rfr']:.3f} ({rfr['repeated_errors']}/{rfr['total_errors']} repeated)")

# 2. Retrieval Precision
#    Fraction of retrieved nodes that the generator actually referenced.
#    Higher means retrieval is focused and relevant.
ret = summary["retrieval_metrics"]
print(f"Retrieval precision: {ret['mean_precision']:.3f}")
print(f"Mean retrieval time: {ret['mean_retrieval_time_s']:.3f}s")

# 3. Concept Coverage
#    Fraction of XBRL entity types with at least one learned strategy.
#    Measures how broadly the graph covers the domain.
print(f"Concept coverage: {summary['concept_coverage']:.2%}")
```

## Output Structure

After a GSAM run, the following files are produced:

```
results/gsam_run_TIMESTAMP_finer_online/
├── run_config.json                     # Full configuration used
├── final_results.json                  # Consolidated accuracy results
├── initial_test_results.json           # Baseline accuracy (empty graph)
├── test_results.json                   # Final accuracy after adaptation
├── retrieval_logs.jsonl                # Per-task retrieval precision and timing
├── error_tracking.jsonl                # Per-error concept and confusion tracking
├── graph_stats.json                    # Final graph summary statistics
├── detailed_llm_logs/                  # Raw LLM request/response logs
└── graph_checkpoints/
    ├── graph_step_0.json               # Initial graph (ontology only)
    ├── graph_step_50.json              # Intermediate checkpoint
    ├── graph_best.json                 # Best validation accuracy (offline only)
    └── graph_final.json                # Final evolved graph
```

### Understanding the graph JSON

Each graph checkpoint is a JSON file you can load and inspect:

```python
from gsam.graph_memory import KnowledgeGraph, NodeType

graph = KnowledgeGraph.load("results/.../graph_checkpoints/graph_final.json")

# Print summary
print(graph)
# KnowledgeGraph(nodes=245, edges=412, concepts=139, coverage=0.67)

# Inspect specific nodes
for nid in graph.get_nodes_by_type(NodeType.STRATEGY)[:5]:
    data = graph.graph.nodes[nid]
    print(f"[{nid}] helpful={data['helpful_count']} harmful={data['harmful_count']}")
    print(f"  {data['content'][:100]}")
```

## CLI Reference

### `eval.finance.run_gsam` -- Full argument list

<details>
<summary>Click to expand</summary>

| Argument | Description | Default |
|----------|-------------|---------|
| `--task_name` | Task name (`finer` or `formula`) | Required |
| `--mode` | `offline`, `online`, or `eval_only` | `offline` |
| `--save_path` | Directory to save results | Required |
| `--api_provider` | `sambanova`, `together`, or `openai` | `sambanova` |
| `--generator_model` | Model for the generator agent | `DeepSeek-V3.1` |
| `--reflector_model` | Model for the reflector agent | `DeepSeek-V3.1` |
| `--curator_model` | Model for the curator agent | `DeepSeek-V3.1` |
| `--num_epochs` | Training epochs (offline only) | `1` |
| `--max_num_rounds` | Max reflection rounds per incorrect answer | `3` |
| `--curator_frequency` | Run curator every N training steps | `1` |
| `--eval_steps` | Evaluate on validation set every N steps | `100` |
| `--online_eval_frequency` | Window size for online test-then-train | `15` |
| `--save_steps` | Save graph checkpoint every N steps | `50` |
| `--max_tokens` | Max tokens per LLM call | `4096` |
| `--playbook_token_budget` | Token budget for graph serialization | `80000` |
| `--test_workers` | Parallel workers for test evaluation | `20` |
| `--json_mode` | Enable JSON mode for LLM calls | `False` |
| `--no_ground_truth` | Omit ground truth from reflector | `False` |
| `--taxonomy_path` | Path to XBRL taxonomy JSON | `./eval/finance/data/xbrl_taxonomy.json` |
| `--merge_threshold` | Cosine similarity for node deduplication | `0.9` |
| `--retrieval_depth` | BFS depth for graph retrieval | `2` |
| `--prune_frequency` | Prune low-utility nodes every N steps | `50` |
| `--no_ontology` | Skip XBRL taxonomy initialization | `False` |
| `--no_failure_cascades` | Skip anti-pattern and failure edge creation | `False` |
| `--embedding_only_retrieval` | Use embedding similarity only (no graph BFS) | `False` |
| `--untyped_edges` | All edges become generic `related_to` | `False` |
| `--max_samples` | Limit samples for smoke testing | `None` |

</details>

## Tests

Run the full test suite:

```bash
python -m unittest tests.test_graph_memory tests.test_metrics tests.test_ontology -v
```

The tests cover:

- **test_graph_memory.py** (36 tests): Node/edge CRUD, auto-ID generation, node merging, subgraph extraction, neighbor filtering, taxonomy helpers (ancestors, children, siblings, LCA), serialization/deserialization, pruning rules, graph statistics
- **test_metrics.py** (14 tests): RFR computation with repeated/unique/confusion-pair errors, retrieval precision aggregation, concept coverage, transfer metric computation (positive/negative/zero delta)
- **test_ontology.py** (7 tests): Taxonomy loading from JSON, category/subcategory/entity node creation, is_a edge structure, taxonomy path attributes, sibling resolution, entity name mapping

## Citation

If you use GSAM or ACE in your research, please cite:

```bibtex
@misc{zhang2025agenticcontextengineeringevolving,
      title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models},
      author={Qizheng Zhang and Changran Hu and Shubhangi Upasani and Boyuan Ma and Fenglu Hong and Vamsidhar Kamanuru and Jay Rainton and Chen Wu and Mengmeng Ji and Hanchen Li and Urmish Thakker and James Zou and Kunle Olukotun},
      year={2025},
      eprint={2510.04618},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04618},
}
```
