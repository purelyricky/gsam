"""
Offline (multi-epoch) training loop for the ACE orchestrator, split out of ace.py
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


class OfflineTrainingMixin:
    """Multi-epoch offline training loop."""

    def _offline_train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        data_processor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str,
        start_epoch: int = 1,
        start_step: int = 0,
    ) -> Dict[str, Any]:
        """
        Run offline training
        
        Args:
            train_samples: List of training samples
            val_samples: List of validation samples
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            save_path: Path to save results
            usage_log_path: Path for bullet usage logging
            playbook_dir: Directory for intermediate playbooks
            log_dir: Directory for detailed logs
            
        Returns:
            Dictionary with training results
        """
        # Extract configuration using helper
        config_params = self._extract_config_params(config)
        task_name = config_params['task_name']
        num_epochs = config_params['num_epochs']
        eval_steps = config_params['eval_steps']
        save_steps = config_params['save_steps']
        test_workers = config_params['test_workers']
        use_json_mode = config_params['use_json_mode']
        curator_frequency = config_params['curator_frequency']
        
        # Initialize tracking
        results = []
        pre_train_post_train_results = []
        error_logs = []
        best_accuracy = 0.0
        self.best_playbook = self.playbook
        all_latencies = []

        logger.info(f"Total epochs: {num_epochs}")
        logger.info(f"Train samples per epoch: {len(train_samples)}")
        logger.info(f"Val samples: {len(val_samples)}")
        logger.info(f"Curator frequency: every {curator_frequency} steps")
        logger.info(f"Evaluation frequency: every {eval_steps} steps\n")
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # --- RESUME: skip epochs already completed ---
            if epoch < start_epoch:
                logger.info(f"Skipping epoch {epoch} (resuming from epoch {start_epoch}, step {start_step})")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch}/{num_epochs}")
            logger.info(f"{'='*60}")

            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []

            for step, task_dict in enumerate(train_samples):
                step += 1

                # --- RESUME: skip steps already completed in the resume epoch ---
                if epoch == start_epoch and step <= start_step:
                    continue

                logger.info(f"\n--- Step {step}/{len(train_samples)} ---")
                
                target = task_dict.get("target", "")

                # Use helper method for training single sample
                try:
                    pre_train_answer, post_train_answer, tracking_dict, latency = self._train_single_sample(
                        task_dict=task_dict,
                        data_processor=data_processor,
                        step_id=f"train_e_{epoch}_s_{step}",
                        epoch=epoch,
                        step=step,
                        usage_log_path=usage_log_path,
                        log_dir=log_dir,
                        config_params=config_params,
                        total_samples=len(train_samples)
                    )
                except Exception as e:
                    logger.info(f"[WARN] Skipping training sample (epoch={epoch}, step={step}): {type(e).__name__}: {e}")
                    self._save_progress(save_path, mode="offline", epoch=epoch, step=step)
                    continue

                # Collect answers for accuracy calculation
                epoch_answers_pre_train.append(pre_train_answer)
                epoch_targets_pre_train.append(target)
                epoch_answers_post_train.append(post_train_answer)
                epoch_targets_post_train.append(target)
                all_latencies.append(latency)

                # Track pre-train and post-train results
                pre_train_post_train_result = {
                    "epoch": epoch,
                    "step": step,
                    "target": target,
                    **tracking_dict
                }
                pre_train_post_train_results.append(pre_train_post_train_result)

                # Save progress cursor so this run can be resumed if interrupted
                self._save_progress(save_path, mode="offline", epoch=epoch, step=step)

                # Save intermediate playbook
                if step % save_steps == 0:
                    intermediate_path = os.path.join(
                        playbook_dir, f"epoch_{epoch}_step_{step}_playbook.txt"
                    )
                    with open(intermediate_path, "w") as f:
                        f.write(self.playbook)
                
                # Periodic evaluation
                if step % eval_steps == 0:
                    logger.info(f"\n{'='*40}")
                    logger.info(f"EVALUATION AT EPOCH {epoch}, STEP {step}")
                    logger.info(f"{'='*40}")
                    
                    # Compute training accuracies
                    pre_train_accuracy = data_processor.evaluate_accuracy(
                        epoch_answers_pre_train, epoch_targets_pre_train
                    )
                    post_train_accuracy = data_processor.evaluate_accuracy(
                        epoch_answers_post_train, epoch_targets_post_train
                    )
                    
                    # Validation evaluation
                    val_results = {}
                    if val_samples:
                        val_results, val_error_log = evaluate_test_set(
                            data_processor, self.generator, self.playbook, 
                            val_samples, self.max_tokens, log_dir, 
                            max_workers=test_workers, use_json_mode=use_json_mode
                        )
                    
                    result = {
                        "epoch": epoch,
                        "step": step,
                        "train_result": {
                            "pre_train_accuracy": pre_train_accuracy,
                            "post_train_accuracy": post_train_accuracy
                        },
                        "val_result": val_results,
                        "playbook_num_tokens": count_tokens(self.playbook),
                        "playbook_length": len(self.playbook),
                        "playbook_stats": get_playbook_stats(self.playbook)
                    }
                    results.append(result)
                    error_logs.append({
                        "epoch": epoch,
                        "step": step,
                        "val_results": val_results,
                        "error_log": val_error_log
                    })

                    # Track best playbook
                    if val_results:
                        acc = val_results["accuracy"]
                        if acc > best_accuracy:
                            best_accuracy = acc
                            self.best_playbook = self.playbook
                            logger.info(f"New best accuracy: {best_accuracy:.3f}")
                    
                    # Save results
                    results_path = os.path.join(save_path, "train_results.json")
                    with open(results_path, "w") as f:
                        json.dump({
                            "best_accuracy": best_accuracy,
                            "results": results,
                        }, f, indent=2)
                    
                    error_logs_path = os.path.join(save_path, "val_results.json")
                    with open(error_logs_path, "w") as f:
                        json.dump(error_logs, f, indent=2)
            
            # End of epoch - save final playbook
            epoch_playbook_path = os.path.join(
                playbook_dir, f"epoch_{epoch}_final_playbook.txt"
            )
            with open(epoch_playbook_path, "w") as f:
                f.write(self.playbook)

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

        # Save training results
        results_path = os.path.join(save_path, "train_results.json")
        with open(results_path, "w") as f:
            json.dump({
                "best_accuracy": best_accuracy,
                "results": results,
            }, f, indent=2)

        pre_train_post_train_results_path = os.path.join(save_path, "pre_train_post_train_results.json")
        with open(pre_train_post_train_results_path, "w") as f:
            json.dump(pre_train_post_train_results, f, indent=2)

        # Save final playbook
        final_playbook_path = os.path.join(save_path, f"final_playbook.txt")
        with open(final_playbook_path, "w") as f:
            f.write(self.playbook)

        # Save best playbook
        best_playbook_path = os.path.join(save_path, f"best_playbook.txt")
        with open(best_playbook_path, "w") as f:
            f.write(self.best_playbook)

        logger.info(f"\n{'='*60}")
        logger.info(f"OFFLINE TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Best Validation Accuracy: {best_accuracy:.3f}")
        logger.info(f"{'='*60}\n")

        return {"best_validation_accuracy": best_accuracy, "latency_stats": latency_stats}
