"""Offline (multi-epoch) and online (sliding-window) training loops for the GSAM orchestrator, split out of gsam.py as a mixin."""

import os
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .graph_memory import KnowledgeGraph, NodeType, EdgeType
from .core.graph_constructor import GraphConstructor
from .core.graph_retriever import GraphRetriever
from .ontology import initialize_ontology, get_entity_name_to_node_map
from .prompts.generator import GSAM_GENERATOR_PROMPT
from .prompts.reflector import GSAM_REFLECTOR_PROMPT, GSAM_REFLECTOR_PROMPT_NO_GT
from .prompts.curator import GSAM_CURATOR_PROMPT, GSAM_CURATOR_PROMPT_NO_GT

from llm import timed_llm_call
from utils import initialize_clients, extract_answer, count_tokens, evaluate_test_set
from logger import log_llm_call
from playbook_utils import extract_json_from_text

from log_config import get_logger
from experiment_tracker import ExperimentTracker
from resume_utils import find_latest_checkpoint, write_progress, default_resume_state, apply_progress_files

logger = get_logger("gsam")


from .sample_training import SampleTrainingMixin
from .core.agents import GSAMGenerator, GSAMReflector, GSAMCurator, _RetrievalAwareGenerator



class TrainingLoopsMixin:
    """Offline and online training loops."""

    def _offline_train(self, train_samples, val_samples, data_processor, config,
                       save_path, log_dir, graph_dir,
                       start_epoch: int = 1, start_step: int = 0):
        """Offline training loop."""
        config_params = self._extract_config_params(config)
        num_epochs = config_params['num_epochs']
        eval_steps = config_params['eval_steps']
        save_steps = config_params['save_steps']

        results = []
        pre_post_results = []
        best_accuracy = 0.0
        all_latencies = []

        for epoch in range(1, num_epochs + 1):
            # --- RESUME: skip epochs already completed ---
            if epoch < start_epoch:
                logger.info(f"Skipping epoch {epoch} (resuming from epoch {start_epoch}, step {start_step})")
                continue

            logger.info(f"\n{'='*60}\nEPOCH {epoch}/{num_epochs}\n{'='*60}")

            answers_pre, targets_pre = [], []
            answers_post, targets_post = [], []

            for step, task_dict in enumerate(train_samples):
                step += 1

                # --- RESUME: skip steps already completed in the resume epoch ---
                if epoch == start_epoch and step <= start_step:
                    continue

                logger.info(f"\n--- Step {step}/{len(train_samples)} ---")

                target = task_dict.get("target", "")
                try:
                    pre, post, tracking, latency = self._train_single_sample(
                        task_dict, data_processor, f"train_e_{epoch}_s_{step}",
                        step, log_dir, config_params, len(train_samples),
                    )
                except Exception as e:
                    logger.info(f"[WARN] Skipping training sample (epoch={epoch}, step={step}): {type(e).__name__}: {e}")
                    self._save_progress(save_path, mode="offline", epoch=epoch, step=step)
                    continue

                answers_pre.append(pre)
                targets_pre.append(target)
                answers_post.append(post)
                targets_post.append(target)
                all_latencies.append(latency)

                pre_post_results.append({
                    "epoch": epoch, "step": step, "target": target, **tracking,
                })

                # Save progress cursor so this run can be resumed if interrupted
                self._save_progress(save_path, mode="offline", epoch=epoch, step=step)

                if step % save_steps == 0:
                    self.knowledge_graph.save(
                        os.path.join(graph_dir, f"graph_epoch_{epoch}_step_{step}.json")
                    )

                if step % eval_steps == 0:
                    pre_acc = data_processor.evaluate_accuracy(answers_pre, targets_pre)
                    post_acc = data_processor.evaluate_accuracy(answers_post, targets_post)

                    val_results = {}
                    if val_samples:
                        retrieval_gen = _RetrievalAwareGenerator(self.generator, self.graph_retriever)
                        val_results, _ = evaluate_test_set(
                            data_processor, retrieval_gen, "",
                            val_samples, self.max_tokens, log_dir,
                            max_workers=config_params['test_workers'],
                            use_json_mode=config_params['use_json_mode'],
                        )

                    result = {
                        "epoch": epoch, "step": step,
                        "train_result": {"pre_train_accuracy": pre_acc, "post_train_accuracy": post_acc},
                        "val_result": val_results,
                        "graph_stats": self.knowledge_graph.stats(),
                    }
                    results.append(result)

                    if val_results and val_results.get("accuracy", 0) > best_accuracy:
                        best_accuracy = val_results["accuracy"]
                        self.knowledge_graph.save(os.path.join(graph_dir, "graph_best.json"))

            # Multi-epoch graph refinement (Paper §5.6)
            # After each epoch, consolidate: discover edges, reinforce
            # weights, merge similar strategies with identical neighborhoods.
            if not config.get('no_multi_epoch_refinement', False):
                consolidation_stats = self.knowledge_graph.consolidate_epoch()
                logger.info(f"Epoch {epoch} consolidation: {consolidation_stats}")
                self.knowledge_graph.save(
                    os.path.join(graph_dir, f"graph_epoch_{epoch}_consolidated.json")
                )

        # Compute latency stats
        latency_stats = {}
        if all_latencies:
            n = len(all_latencies)
            for key in ("generator_time", "reflector_time", "curator_time",
                        "retrieval_time", "graph_update_time"):
                latency_stats[key.replace("_time", "_mean_s")] = sum(
                    d[key] for d in all_latencies
                ) / n
            latency_stats["total_per_sample_mean_s"] = sum(
                sum(d.values()) for d in all_latencies
            ) / n

        # Save results
        with open(os.path.join(save_path, "train_results.json"), "w") as f:
            json.dump({"best_accuracy": best_accuracy, "results": results}, f, indent=2)
        with open(os.path.join(save_path, "pre_train_post_train_results.json"), "w") as f:
            json.dump(pre_post_results, f, indent=2)

        return {"best_validation_accuracy": best_accuracy, "latency_stats": latency_stats}

    def _online_train_and_test(self, test_samples, data_processor, config,
                               save_path, log_dir, graph_dir,
                               start_window: int = 0, start_global_step: int = 0,
                               prior_online_results: Optional[Dict] = None):
        """Online training and testing loop."""
        config_params = self._extract_config_params(config)
        online_eval_frequency = config.get('online_eval_frequency', 15)
        save_steps = config_params['save_steps']

        # Initialize accumulators — restore from saved partial results when resuming
        if prior_online_results:
            all_answers = prior_online_results.get('all_answers', [])
            all_targets = prior_online_results.get('all_targets', [])
            window_results = prior_online_results.get('window_results', [])
            correct_count_sample = prior_online_results.get('correct_count_sample', 0)
            total_count = prior_online_results.get('total_count', 0)
            skipped_count = prior_online_results.get('skipped_count', 0)
            global_step = start_global_step
            logger.info(f"Resumed online: {len(window_results)} windows already done, "
                  f"global_step={global_step}")
        else:
            correct_count_sample = 0
            total_count = 0
            skipped_count = 0
            all_answers = []
            all_targets = []
            window_results = []
            global_step = 0

        all_errors = []
        all_latencies = []

        num_windows = (len(test_samples) + online_eval_frequency - 1) // online_eval_frequency

        for window_idx in range(num_windows):
            # --- RESUME: skip windows already completed ---
            if window_idx < start_window:
                continue

            start = window_idx * online_eval_frequency
            end = min((window_idx + 1) * online_eval_frequency, len(test_samples))
            window_samples = test_samples[start:end]

            logger.info(f"\n{'='*60}\nWINDOW {window_idx+1}/{num_windows} (samples {start}-{end-1})\n{'='*60}")

            # Test on window with per-query retrieval
            retrieval_gen = _RetrievalAwareGenerator(self.generator, self.graph_retriever)
            w_results, w_errors = evaluate_test_set(
                data_processor, retrieval_gen, "",
                window_samples, self.max_tokens, log_dir,
                max_workers=config_params['test_workers'],
                use_json_mode=config_params['use_json_mode'],
            )

            w_acc = w_results['accuracy']
            w_correct = w_results['correct']
            w_total = w_results['total']
            w_skipped = w_results.get('skipped', 0)
            correct_count_sample += w_correct
            total_count += w_total
            skipped_count += w_skipped
            # Accumulate raw answers/targets so final accuracy can be
            # computed via data_processor.evaluate_accuracy (which uses
            # the correct metric — token-level for FiNER, sample-level
            # for Formula) rather than multiplying window accuracy by
            # window count, which conflates metric levels.
            all_answers.extend(w_results.get('answers', []))
            all_targets.extend(w_results.get('targets', []))

            for err in w_errors.get('errors', []):
                all_errors.append({
                    "window": window_idx + 1,
                    "global_index": start + err['index'],
                    "prediction": err['prediction'],
                    "ground_truth": err['ground_truth'],
                })

            window_results.append({
                "window": window_idx + 1,
                "start_idx": start, "end_idx": end,
                "window_accuracy": w_acc,
                "window_correct": w_correct, "window_total": w_total,
            })

            cum_acc = data_processor.evaluate_accuracy(all_answers, all_targets) if all_answers else 0
            logger.info(f"Window {window_idx+1} accuracy: {w_acc:.3f} | Cumulative: {cum_acc:.3f}")

            # Train on window
            for local_step, task_dict in enumerate(window_samples):
                global_step += 1
                logger.info(f"\n--- Window {window_idx+1}, Step {local_step+1}/{len(window_samples)} (Global {global_step}) ---")

                try:
                    _, _, _, latency = self._train_single_sample(
                        task_dict, data_processor, f"online_train_s_{global_step}",
                        global_step, log_dir, config_params, len(test_samples),
                    )
                    all_latencies.append(latency)
                except Exception as e:
                    logger.info(f"[WARN] Skipping online training step {global_step}: {type(e).__name__}: {e}")

                if global_step % save_steps == 0:
                    self.knowledge_graph.save(
                        os.path.join(graph_dir, f"graph_step_{global_step}.json")
                    )

            # Save partial results and progress after each window so the run
            # can be resumed from the next window if interrupted.
            partial = {
                'all_answers': all_answers,
                'all_targets': all_targets,
                'window_results': window_results,
                'correct_count_sample': correct_count_sample,
                'total_count': total_count,
                'skipped_count': skipped_count,
            }
            with open(os.path.join(save_path, "partial_online_results.json"), "w") as _pf:
                json.dump(partial, _pf)
            self._save_progress(save_path, mode="online",
                                window=window_idx, global_step=global_step)

        # Compute final accuracy using the proper per-task metric
        if all_answers and all_targets:
            final_acc = data_processor.evaluate_accuracy(all_answers, all_targets)
        else:
            final_acc = 0.0

        # Compute latency stats
        latency_stats = {}
        if all_latencies:
            n = len(all_latencies)
            for key in ("generator_time", "reflector_time", "curator_time",
                        "retrieval_time", "graph_update_time"):
                latency_stats[key.replace("_time", "_mean_s")] = sum(
                    d[key] for d in all_latencies
                ) / n
            latency_stats["total_per_sample_mean_s"] = sum(
                sum(d.values()) for d in all_latencies
            ) / n

        if skipped_count > 0:
            logger.warning(f"Warning: {skipped_count} sample(s) skipped due to LLM failures and excluded from accuracy.")

        # Save results
        with open(os.path.join(save_path, "test_results.json"), "w") as f:
            json.dump({
                "test_accuracy": final_acc,
                "skipped": skipped_count,
                "window_results": window_results,
                "errors": all_errors,
            }, f, indent=2)

        logger.info(f"\nFinal Online Test Accuracy: {final_acc:.3f} (skipped={skipped_count})")

        return {
            "accuracy": final_acc,
            "correct": correct_count_sample,
            "total": total_count,
            "skipped": skipped_count,
            "latency_stats": latency_stats,
        }
