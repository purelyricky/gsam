# Reproducibility Notes

- **Configs**: every experiment is described by a JSON file in `experiments/configs/` (tasks, ablations, models, retrieval settings).
  Run one with `python -m experiments.run_experiment --config <file> --save_path results`.
- **Seeds**: FiNER transfer splits are built with `random.seed(42)` in `eval/finance/finer_transfer.py`.
- **Run artifacts**: each run writes `run_config.json` (exact config used), `final_results.json`, `graph_stats.json`,
  `progress.json` (resume checkpoint) and a log file under `results/<run>/`. Archived runs live in `clean_results/`.
- **Resume**: interrupted runs resume from the latest checkpoint in `progress.json` / graph checkpoints.
- **Error analysis**: `analyze_results.py` aggregates per-run metrics and ablation deltas; `gsam/metrics.py` computes
  accuracy, per-class F1 and failure breakdowns; `bullet_usage_log.jsonl` and `curator_operations_diff.jsonl` record
  which memory items influenced each prediction.
- **Offline verification**: the unit tests (`python -m unittest discover -s tests`) stub all LLM calls and disable embedding
  downloads, so they run without API keys or network access. Full experiments require a provider key (see `.env.example`).
- **Containers**: `docker compose up --build` runs the test suite in a clean image;
  `docker compose --profile experiment run experiment` runs experiments with your `.env`.
- **Logging**: set `GSAM_LOG_FORMAT=json` for machine-readable logs that can be shipped to wandb/mlflow or any log store.
