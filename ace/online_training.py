"""
Online (sliding-window train-and-test) loop for the ACE orchestrator, split out of ace.py
as a mixin so the orchestrator module stays focused on setup and the shared steps.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from playbook_utils import *
from logger import *
from utils import *

from log_config import get_logger

logger = get_logger("ace")


class OnlineTrainingMixin:
    """Sliding-window online train-and-test loop."""

    def _online_train_and_test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str,
        start_window: int = 0,
        start_global_step: int = 0,
        prior_online_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run online training and testing
        
        Args:
            test_samples: List of samples to train and test on
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            save_path: Path to save results
            usage_log_path: Path for bullet usage logging
            playbook_dir: Directory for intermediate playbooks
            log_dir: Directory for detailed logs
            
        Returns:
            Dictionary with training results, test results, and final playbook
        """
        # Extract configuration using helper
        config_params = self._extract_config_params(config)
        num_epochs = config_params['num_epochs']
        
        # Validate configuration
        if num_epochs != 1:
            raise ValueError(f"online_train_and_test requires num_epochs=1, got {num_epochs}")
        
        # Extract additional parameters
        curator_frequency = config_params['curator_frequency']
        task_name = config_params['task_name']
        save_steps = config_params['save_steps']
        use_json_mode = config_params['use_json_mode']
        test_workers = config_params['test_workers']
        online_eval_frequency = config.get('online_eval_frequency', 100)  # Get from config
        
        # Initialize tracking
        train_results = []
        pre_train_post_train_results = []
        all_latencies = []
        all_test_errors = []

        # Test tracking - restore from saved partial results when resuming
        if prior_online_results:
            correct_count_sample_based = prior_online_results.get('correct_count_sample_based', 0)
            correct_count = prior_online_results.get('correct_count', 0)
            total_count = prior_online_results.get('total_count', 0)
            skipped_count = prior_online_results.get('skipped_count', 0)
            window_test_results = prior_online_results.get('window_test_results', [])
            logger.info(f"Resumed online: {len(window_test_results)} windows done, "
                  f"global_step={start_global_step}")
        else:
            correct_count_sample_based = 0
            correct_count = 0
            total_count = 0
            skipped_count = 0
            window_test_results = []
        logger.info(f"Total samples: {len(test_samples)}")
        logger.info(f"Window size: {online_eval_frequency}")
        logger.info(f"Number of windows: {(len(test_samples) + online_eval_frequency - 1) // online_eval_frequency}")
        logger.info(f"Curator frequency: every {curator_frequency} steps")
        
        # Split samples into windows
        num_windows = (len(test_samples) + online_eval_frequency - 1) // online_eval_frequency
        
        epoch = 1  # Always 1 epoch
        global_step = start_global_step

        for window_idx in range(num_windows):
            # --- RESUME: skip windows already completed ---
            if window_idx < start_window:
                continue

            start_idx = window_idx * online_eval_frequency
            end_idx = min((window_idx + 1) * online_eval_frequency, len(test_samples))
            window_samples = test_samples[start_idx:end_idx]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"WINDOW {window_idx + 1}/{num_windows}")
            logger.info(f"Samples {start_idx} to {end_idx - 1}")
            logger.info(f"{'='*60}")
            
            # =================================================================
            # STEP 1: TEST on window with current playbook (before training)
            # =================================================================
            logger.info(f"\n--- Testing window {window_idx + 1} with current playbook ---")
            
            # Use evaluate_test_set for parallel evaluation
            window_test_results_dict, window_test_error_log = evaluate_test_set(
                data_processor,
                self.generator,
                self.playbook,
                window_samples,
                self.max_tokens,
                log_dir,
                max_workers=test_workers,
                use_json_mode=use_json_mode
            )
            
            # Extract results
            window_accuracy = window_test_results_dict['accuracy']
            window_correct = window_test_results_dict['correct']
            window_total = window_test_results_dict['total']
            window_skipped = window_test_results_dict.get('skipped', 0)
            correct_count_sample_based += window_correct
            correct_count += window_accuracy * window_total
            total_count += window_total
            skipped_count += window_skipped
            
            # Add errors with window and global index information
            for error in window_test_error_log['errors']:
                all_test_errors.append({
                    "window": window_idx + 1,
                    "global_index": start_idx + error['index'],
                    "prediction": error['prediction'],
                    "ground_truth": error['ground_truth']
                })
            
            window_test_results.append({
                "window": window_idx + 1,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "window_accuracy": window_accuracy,
                "window_correct": window_correct,
                "window_total": window_total
            })
            
            # Calculate cumulative test accuracy so far
            cumulative_test_accuracy = correct_count / total_count
            
            logger.info(f"Window {window_idx + 1} test accuracy: {window_accuracy:.3f}")
            logger.info(f"Cumulative test accuracy so far: {cumulative_test_accuracy:.3f} "
                  f"({total_count} samples)")
            
            # =================================================================
            # STEP 2: TRAIN on window (same as offline_train)
            # =================================================================
            logger.info(f"\n--- Training on window {window_idx + 1} ---")
            
            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []
            
            for local_step, task_dict in enumerate(window_samples):
                global_step += 1
                local_step += 1
                
                logger.info(f"\n--- Window {window_idx + 1}, Step {local_step}/{len(window_samples)} "
                      f"(Global step {global_step}) ---")
                
                target = task_dict.get("target", "")

                # Use helper method for training single sample
                try:
                    pre_train_answer, post_train_answer, tracking_dict, latency = self._train_single_sample(
                        task_dict=task_dict,
                        data_processor=data_processor,
                        step_id=f"online_train_s_{global_step}",
                        epoch=epoch,
                        step=global_step,
                        usage_log_path=usage_log_path,
                        log_dir=log_dir,
                        config_params=config_params,
                        total_samples=len(test_samples)
                    )
                except Exception as e:
                    logger.info(f"[WARN] Skipping online training step {global_step}: {type(e).__name__}: {e}")
                    continue

                # Collect answers for accuracy calculation
                epoch_answers_pre_train.append(pre_train_answer)
                epoch_targets_pre_train.append(target)
                epoch_answers_post_train.append(post_train_answer)
                epoch_targets_post_train.append(target)
                all_latencies.append(latency)

                # Track pre-train and post-train results
                pre_train_post_train_result = {
                    "window": window_idx + 1,
                    "global_step": global_step,
                    "target": target,
                    **tracking_dict
                }
                pre_train_post_train_results.append(pre_train_post_train_result)
                
                # Save intermediate playbook
                if global_step % save_steps == 0:
                    intermediate_path = os.path.join(
                        playbook_dir, f"step_{global_step}_playbook.txt"
                    )
                    with open(intermediate_path, "w") as f:
                        f.write(self.playbook)
            
            # End of window - compute training accuracies for this window
            pre_train_accuracy = data_processor.evaluate_accuracy(
                epoch_answers_pre_train, epoch_targets_pre_train
            )
            post_train_accuracy = data_processor.evaluate_accuracy(
                epoch_answers_post_train, epoch_targets_post_train
            )
            
            window_train_result = {
                "window": window_idx + 1,
                "global_step": global_step,
                "train_result": {
                    "pre_train_accuracy": pre_train_accuracy,
                    "post_train_accuracy": post_train_accuracy
                },
                "cumulative_test_accuracy": cumulative_test_accuracy,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook),
                "playbook_stats": get_playbook_stats(self.playbook)
            }
            train_results.append(window_train_result)
            
            # Save partial results + progress cursor after each window
            partial = {
                'correct_count_sample_based': correct_count_sample_based,
                'correct_count': correct_count,
                'total_count': total_count,
                'skipped_count': skipped_count,
                'window_test_results': window_test_results,
            }
            with open(os.path.join(save_path, "partial_online_results.json"), "w") as _pf:
                json.dump(partial, _pf)
            self._save_progress(save_path, mode="online",
                                window=window_idx, global_step=global_step)

            logger.info(f"\nWindow {window_idx + 1} training complete:")
            logger.info(f"  Pre-train accuracy: {pre_train_accuracy:.3f}")
            logger.info(f"  Post-train accuracy: {post_train_accuracy:.3f}")
            
            # Save window playbook
            window_playbook_path = os.path.join(
                playbook_dir, f"window_{window_idx + 1}_final_playbook.txt"
            )
            with open(window_playbook_path, "w") as f:
                f.write(self.playbook)
        
        # All windows complete
        logger.info(f"\n{'='*60}")
        logger.info(f"ONLINE TRAIN AND TEST COMPLETE")
        logger.info(f"{'='*60}")

        # Calculate final cumulative test accuracy
        # total_count may be < len(test_samples) when samples are skipped due
        # to LLM failures (empty/truncated responses); that is expected behavior.
        if skipped_count > 0:
            logger.warning(f"Warning: {skipped_count} sample(s) skipped due to LLM failures and excluded from accuracy.")
        final_test_accuracy = correct_count / total_count if total_count > 0 else 0.0

        # Compute latency stats
        latency_stats = {}
        if all_latencies:
            n = len(all_latencies)
            for key in ("generator_time", "reflector_time", "curator_time",
                        "playbook_update_time"):
                latency_stats[key.replace("_time", "_mean_s")] = sum(
                    d[key] for d in all_latencies
                ) / n
            latency_stats["retrieval_mean_s"] = 0.0
            latency_stats["graph_update_mean_s"] = 0.0
            latency_stats["total_per_sample_mean_s"] = sum(
                sum(d.values()) for d in all_latencies
            ) / n

        test_results = {
            "accuracy": final_test_accuracy,
            "correct": correct_count_sample_based,
            "total": total_count,
            "skipped": skipped_count,
            "window_results": window_test_results
        }

        test_error_log = {
            "accuracy": final_test_accuracy,
            "errors": all_test_errors
        }

        # Save test results
        test_results_path = os.path.join(save_path, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump({
                "test_accuracy": final_test_accuracy,
                "test_results": test_results,
                "test_error_log": test_error_log
            }, f, indent=2)

        # Save training results (per window)
        train_results_path = os.path.join(save_path, "train_results.json")
        with open(train_results_path, "w") as f:
            json.dump({"train_results": train_results}, f, indent=2)

        # Save pre-train/post-train results
        pre_train_post_train_results_path = os.path.join(save_path, "pre_train_post_train_results.json")
        with open(pre_train_post_train_results_path, "w") as f:
            json.dump(pre_train_post_train_results, f, indent=2)

        # Save final playbook
        final_playbook_path = os.path.join(save_path, f"final_playbook.txt")
        with open(final_playbook_path, "w") as f:
            f.write(self.playbook)

        logger.info(f"\n{'='*60}")
        logger.info(f"ONLINE TRAINING AND TESTING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Final Test Accuracy: {final_test_accuracy:.3f}")
        logger.info(f"{'='*60}\n")

        return {
            "accuracy": final_test_accuracy,
            "correct": correct_count_sample_based,
            "total": total_count,
            "latency_stats": latency_stats,
        }
