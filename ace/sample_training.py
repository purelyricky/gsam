"""Single-sample generate / reflect / curate step for the ACE orchestrator, split out of ace.py as a mixin."""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .core import Generator, Reflector, Curator, BulletpointAnalyzer
from .training_modes import OfflineTrainingMixin
from .online_training import OnlineTrainingMixin
from playbook_utils import *
from logger import *
from utils import *

from log_config import get_logger
from experiment_tracker import ExperimentTracker
from resume_utils import find_latest_checkpoint, write_progress, default_resume_state, apply_progress_files

logger = get_logger("ace")



class SampleTrainingMixin:

    def _train_single_sample(
        self,
        task_dict: Dict[str, Any],
        data_processor,
        step_id: str,
        epoch: int,
        step: int,
        usage_log_path: str,
        log_dir: str,
        config_params: Dict[str, Any],
        total_samples: int
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Train on a single sample with reflection and curation.
        
        Args:
            task_dict: Sample dictionary with question, context, target
            data_processor: Data processor for evaluation
            step_id: Identifier string for this step (e.g., "train_e_1_s_10" or "online_train_w_1_s_5")
            epoch: Current epoch number
            step: Current step number
            usage_log_path: Path for bullet usage logging
            log_dir: Path for logging directory
            config_params: Configuration parameters dictionary
            total_samples: Total number of samples in dataset
            
        Returns:
            Tuple of (pre_train_answer, post_train_answer, tracking_dict, latency_dict)
        """
        # Extract configuration
        max_num_rounds = config_params['max_num_rounds']
        curator_frequency = config_params['curator_frequency']
        token_budget = config_params['token_budget']
        use_json_mode = config_params['use_json_mode']
        no_ground_truth = config_params['no_ground_truth']

        # Extract sample data
        question = task_dict.get("question", "")
        context = task_dict.get("context", "")
        target = task_dict.get("target", "")

        # Per-sample latency accumulators
        _generator_time = 0.0
        _reflector_time = 0.0
        _curator_time = 0.0
        _playbook_update_time = 0.0

        # STEP 1: Initial generation (pre-train)
        logger.info("Generating initial answer...")
        _t0 = time.time()
        gen_response, bullet_ids, call_info = self.generator.generate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_gen_initial",
            log_dir=log_dir
        )
        _generator_time += time.time() - _t0
        
        # Extract answer and check correctness
        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)
        pre_train_answer = final_answer
        
        logger.info(f"Correct: {is_correct}")
        
        # Log bullet usage
        log_bullet_usage(usage_log_path, epoch, step, task_dict, bullet_ids,
                       playbook=self.playbook, is_correct=is_correct)
        
        # Track pre-train result
        tracking_dict = {
            "pre_train_result": {
                "final_answer": final_answer,
                "is_correct": is_correct,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook)
            }
        }
        
        reflection_content = "(empty)"
        
        # STEP 2: Reflection and regeneration
        if not is_correct:
            # For incorrect answers - iterate reflection rounds
            for round_num in range(max_num_rounds):
                logger.info(f"Reflection round {round_num + 1}/{max_num_rounds}")

                # Get bullets for reflector
                playbook_bullets = extract_playbook_bullets(
                    self.playbook, bullet_ids
                )

                # Reflect on error
                _t0 = time.time()
                reflection_content, bullet_tags, _ = self.reflector.reflect(
                    question=question,
                    reasoning_trace=gen_response,
                    predicted_answer=final_answer,
                    ground_truth=target if not no_ground_truth else None,
                    environment_feedback="Predicted answer does not match ground truth",
                    bullets_used=playbook_bullets,
                    use_ground_truth=not no_ground_truth,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_round_{round_num}",
                    log_dir=log_dir
                )
                _reflector_time += time.time() - _t0

                # Update bullet counts
                if bullet_tags:
                    _t0 = time.time()
                    self.playbook = update_bullet_counts(
                        self.playbook, bullet_tags
                    )
                    _playbook_update_time += time.time() - _t0

                # Regenerate with reflection
                _t0 = time.time()
                gen_response, bullet_ids, _ = self.generator.generate(
                    question=question,
                    playbook=self.playbook,
                    context=context,
                    reflection=reflection_content,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_post_reflect_round_{round_num}",
                    log_dir=log_dir
                )
                _generator_time += time.time() - _t0

                final_answer = extract_answer(gen_response)

                if data_processor.answer_is_correct(final_answer, target):
                    logger.info(f"Corrected after reflection round {round_num + 1}!")
                    is_correct = True
                    break

        else:
            # For correct answers - still run reflector to tag helpful bullets
            playbook_bullets = extract_playbook_bullets(
                self.playbook, bullet_ids
            )

            _t0 = time.time()
            reflection_content, bullet_tags, _ = self.reflector.reflect(
                question=question,
                reasoning_trace=gen_response,
                predicted_answer=final_answer,
                ground_truth=target if not no_ground_truth else None,
                environment_feedback="Predicted answer matches ground truth",
                bullets_used=playbook_bullets,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=f"{step_id}_reflect_on_correct",
                log_dir=log_dir
            )
            _reflector_time += time.time() - _t0

            # Update bullet counts
            if bullet_tags:
                _t0 = time.time()
                self.playbook = update_bullet_counts(
                    self.playbook, bullet_tags
                )
                _playbook_update_time += time.time() - _t0

            # Log with reflection
            log_bullet_usage(usage_log_path, epoch, step, task_dict, bullet_ids,
                           playbook=self.playbook,
                           reflection_content=reflection_content,
                           is_correct=is_correct)

        # STEP 3: Curator - Periodically update playbook
        if step % curator_frequency == 0:
            logger.info(f"\n--- Running Curator at step {step} ---")

            stats = get_playbook_stats(self.playbook)

            _t0 = time.time()
            self.playbook, self.next_global_id, operations, _ = self.curator.curate(
                current_playbook=self.playbook,
                recent_reflection=reflection_content,
                question_context=context,
                current_step=step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=step_id,
                log_dir=log_dir,
                next_global_id=self.next_global_id
            )
            _curator_time += time.time() - _t0

            # Run bulletpoint analyzer if enabled
            if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
                logger.info(f"  Running BulletpointAnalyzer (threshold={self.bulletpoint_analyzer_threshold})...")
                _t0 = time.time()
                self.playbook = self.bulletpoint_analyzer.analyze(
                    playbook=self.playbook,
                    threshold=self.bulletpoint_analyzer_threshold,
                    merge=True
                )
                _playbook_update_time += time.time() - _t0

        # STEP 4: Post-curator generation
        _t0 = time.time()
        gen_response, _, _ = self.generator.generate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_post_curate",
            log_dir=log_dir
        )
        _generator_time += time.time() - _t0

        final_answer = extract_answer(gen_response)
        post_train_answer = final_answer

        post_train_is_correct = data_processor.answer_is_correct(final_answer, target)
        tracking_dict["post_train_result"] = {
            "final_answer": final_answer,
            "is_correct": post_train_is_correct,
            "playbook_num_tokens": count_tokens(self.playbook),
            "playbook_length": len(self.playbook)
        }

        latency_dict = {
            "generator_time": _generator_time,
            "reflector_time": _reflector_time,
            "curator_time": _curator_time,
            "playbook_update_time": _playbook_update_time,
        }

        return pre_train_answer, post_train_answer, tracking_dict, latency_dict
