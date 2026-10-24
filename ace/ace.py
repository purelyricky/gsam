"""
ACE (Agent-Curator-Environment) System
Main orchestrator class for training and testing with playbook-based learning.

This module coordinates three agents:
- Generator: Produces answers using playbook knowledge
- Reflector: Analyzes outputs and tags bullets
- Curator: Updates the playbook based on feedback
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .core import Generator, Reflector, Curator, BulletpointAnalyzer
from .training_modes import OfflineTrainingMixin, OnlineTrainingMixin
from playbook_utils import *
from logger import *
from utils import *

from log_config import get_logger
from experiment_tracker import ExperimentTracker
from resume_utils import find_latest_checkpoint, write_progress, default_resume_state, apply_progress_files

logger = get_logger("ace")


class ACE(OfflineTrainingMixin, OnlineTrainingMixin):
    """
    Main ACE system orchestrator.
    
    Manages the training loop where:
    1. Generator produces answers using playbook
    2. Reflector analyzes answers and tags bullets
    3. Curator updates playbook based on feedback
    
    """
    
    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        reflector_model: str,
        curator_model: str,
        max_tokens: int = 4096,
        initial_playbook: Optional[str] = None,
        use_bulletpoint_analyzer: bool = False,
        bulletpoint_analyzer_threshold: float = 0.90
    ):
        """
        Initialize the ACE system.
        
        Args:
            api_provider: API provider for LLM calls
            generator_model: Model name for generator
            reflector_model: Model name for reflector
            curator_model: Model name for curator
            max_tokens: Maximum tokens for LLM calls
            initial_playbook: Initial playbook content (optional)
            use_bulletpoint_analyzer: Whether to use bulletpoint analyzer for deduplication
            bulletpoint_analyzer_threshold: Similarity threshold for bulletpoint analyzer (0-1)
        """
        # Initialize API clients
        generator_client, reflector_client, curator_client = initialize_clients(api_provider)

        # Initialize the three agents
        self.generator = Generator(generator_client, api_provider, generator_model, max_tokens)
        self.reflector = Reflector(reflector_client, api_provider, reflector_model, max_tokens)
        self.curator = Curator(curator_client, api_provider, curator_model, max_tokens)
        
        # Initialize bulletpoint analyzer if requested and available
        self.use_bulletpoint_analyzer = use_bulletpoint_analyzer
        self.bulletpoint_analyzer_threshold = bulletpoint_analyzer_threshold
        
        if use_bulletpoint_analyzer:
            self.bulletpoint_analyzer = BulletpointAnalyzer(
                curator_client, 
                curator_model, 
                max_tokens
            )
            logger.info(f"BulletpointAnalyzer initialized (threshold={bulletpoint_analyzer_threshold})")
        else:
            self.bulletpoint_analyzer = None
        
        # Store configuration
        self.generator_client = generator_client
        self.reflector_client = reflector_client
        self.curator_client = curator_client
        self.max_tokens = max_tokens
        
        # Initialize playbook
        if initial_playbook:
            self.playbook = initial_playbook
        else:
            self.playbook = self._initialize_empty_playbook()
        
        self.best_playbook = self.playbook
        # Track global bullet ID
        self.next_global_id = 1

    # ------------------------------------------------------------------
    # Resume helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_latest_playbook(playbook_dir: str) -> Optional[str]:
        """Return path to the intermediate playbook with the highest step number."""
        return find_latest_checkpoint(playbook_dir, '.txt')

    def _save_progress(self, save_path: str, **kwargs) -> None:
        """Write progress.json so runs can be resumed after interruption."""
        write_progress(save_path, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"), **kwargs)

    def _load_resume_state(self, resume_path: str) -> Dict[str, Any]:
        """
        Load playbook checkpoint and progress from a partial run.
        Updates self.playbook in-place from the latest checkpoint.
        Returns a dict with start_epoch, start_step, start_window,
        start_global_step, prior_online_results.
        """
        state = default_resume_state()
        playbook_dir = os.path.join(resume_path, "intermediate_playbooks")
        latest_pb = self._find_latest_playbook(playbook_dir)
        if latest_pb:
            with open(latest_pb) as f:
                self.playbook = f.read()
            self.best_playbook = self.playbook
            logger.info(f"Loaded playbook checkpoint: {latest_pb}")

        return apply_progress_files(resume_path, state, logger)

    def _initialize_empty_playbook(self) -> str:
        """Initialize an empty playbook with standard sections."""
        return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""
    
    def _extract_config_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract common configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with extracted parameters
        """
        return {
            'num_epochs': config.get('num_epochs', 1),
            'max_num_rounds': config.get('max_num_rounds', 3),
            'curator_frequency': config.get('curator_frequency', 1),
            'eval_steps': config.get('eval_steps', 100),
            'save_steps': config.get('save_steps', 50),
            'token_budget': config.get('playbook_token_budget', 80000),
            'task_name': config.get('task_name', 'default'),
            'use_json_mode': config.get('json_mode', False),
            'no_ground_truth': config.get('no_ground_truth', False),
            'save_dir': config.get('save_dir', './results'),
            'test_workers': config.get('test_workers', 20),
            'use_bulletpoint_analyzer': config.get('use_bulletpoint_analyzer', False),
            'bulletpoint_analyzer_threshold': config.get('bulletpoint_analyzer_threshold', 0.90)
        }
    
    def _setup_paths(self, save_dir: str, task_name: str, mode: str) -> Tuple[str, str]:
        """
        Setup logging paths and directories.
        
        Args:
            save_dir: Base path for saving results
            task_name: task name
            mode: 'offline', 'online', or 'eval_only'
            
        Returns:
            Tuple of (usage_log_path, playbook_dir)
        """
        # Create timestamped run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"ace_run_{timestamp}_{task_name}_{mode}"
        save_path = os.path.join(save_dir, run_folder)
        os.makedirs(save_path, exist_ok=True)
        log_dir = os.path.join(save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        if mode == "eval_only":
            return save_path, log_dir

        usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")
        playbook_dir = os.path.join(save_path, "intermediate_playbooks")
        os.makedirs(playbook_dir, exist_ok=True)
        
        return save_path, usage_log_path, playbook_dir, log_dir
    
    def run(
        self,
        mode: str,
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
        test_samples: Optional[List[Dict[str, Any]]] = None,
        data_processor = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main entrypoint for running ACE system in different modes.
        
        Args:
            mode: Run mode - 'offline', 'online', or 'eval_only'
            train_samples: Training samples (required for offline mode)
            val_samples: Validation samples (required for offline mode)
            test_samples: Test samples (required for online and eval_only modes)
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            
        Returns:
            Dictionary with results depending on the mode
        """
        # Validate inputs
        if mode not in ['offline', 'online', 'eval_only']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'offline', 'online', or 'eval_only'")
        
        if mode == 'offline' and (train_samples is None or val_samples is None):
            raise ValueError("Offline mode requires train_samples and val_samples")
        
        if mode == 'online' and test_samples is None:
            raise ValueError("Online mode requires test_samples")
        
        if mode == 'eval_only' and test_samples is None:
            raise ValueError("eval_only mode requires test_samples")
        
        # Extract configuration
        config_params = self._extract_config_params(config)
        task_name = config_params['task_name']
        save_dir = config_params['save_dir']

        # Determine save_path: resume into existing dir OR create a new timestamped one
        resume_cfg = config.get('resume_path') if config else None
        start_epoch, start_step, start_window, start_global_step = 1, 0, 0, 0
        prior_online_results = None

        if resume_cfg and os.path.isdir(resume_cfg):
            logger.info(f"\nRESUMING ACE run from: {resume_cfg}")
            # Temporarily set save paths from resume dir, then load state
            save_path = resume_cfg
            log_dir = os.path.join(save_path, "detailed_llm_logs")
            os.makedirs(log_dir, exist_ok=True)
            if mode != 'eval_only':
                usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")
                playbook_dir = os.path.join(save_path, "intermediate_playbooks")
                os.makedirs(playbook_dir, exist_ok=True)
            else:
                usage_log_path = None
                playbook_dir = None
            state = self._load_resume_state(resume_cfg)
            start_epoch = state["start_epoch"]
            start_step = state["start_step"]
            start_window = state["start_window"]
            start_global_step = state["start_global_step"]
            prior_online_results = state["prior_online_results"]
        else:
            # Setup paths based on mode (creates new timestamped dir)
            if mode == 'eval_only':
                save_path, log_dir = self._setup_paths(save_dir, task_name, mode)
                usage_log_path = None
                playbook_dir = None
            else:
                save_path, usage_log_path, playbook_dir, log_dir = self._setup_paths(
                    save_dir, task_name, mode)

        # Save configuration
        config_path = os.path.join(save_path, "run_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "task_name": task_name,
                "mode": mode,
                "generator_model": self.generator.model,
                "reflector_model": self.reflector.model,
                "curator_model": self.curator.model,
                "config": config,
            }, f, indent=2)
        
        # Print initial banner
        logger.info(f"\n{'='*60}")
        logger.info(f"ACE SYSTEM - {mode.upper().replace('_', ' ')} MODE")
        logger.info(f"{'='*60}")
        logger.info(f"Task: {task_name}")
        if mode == 'offline':
            logger.info(f"Train samples: {len(train_samples)}")
            logger.info(f"Validation samples: {len(val_samples)}")
            if test_samples:
                logger.info(f"Test samples: {len(test_samples)}")
        elif mode == 'online':
            logger.info(f"Test samples (used for training and testing): {len(test_samples)}")
        else:  # eval_only
            logger.info(f"Test samples: {len(test_samples)}")
        logger.info(f"{'='*60}\n")
        
        # Execute based on mode
        results = {}
        
        if mode == 'offline':
            # OFFLINE MODE WORKFLOW
            # 1. Run initial test if test_samples provided
            if test_samples:
                logger.info(f"\n{'='*60}")
                logger.info(f"INITIAL TEST (before training)")
                logger.info(f"{'='*60}\n")
                initial_test_results = self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=self.playbook,
                    config=config,
                    log_dir=log_dir,
                    save_path=save_path,
                    prefix="initial"
                )
                results['initial_test_results'] = initial_test_results
                logger.info(f"Initial Test Accuracy: {initial_test_results['accuracy']:.3f}\n")

            # 2. Run offline training
            logger.info(f"\n{'='*60}")
            logger.info(f"STARTING OFFLINE TRAINING")
            logger.info(f"{'='*60}\n")
            training_results = self._offline_train(
                train_samples=train_samples,
                val_samples=val_samples,
                data_processor=data_processor,
                config=config,
                save_path=save_path,
                usage_log_path=usage_log_path,
                playbook_dir=playbook_dir,
                log_dir=log_dir,
                start_epoch=start_epoch,
                start_step=start_step,
            )
            results['training_results'] = training_results
            if training_results.get("latency_stats"):
                results['latency_stats'] = training_results["latency_stats"]

            # 3. Run final test if test_samples provided
            if test_samples:
                logger.info(f"\n{'='*60}")
                logger.info(f"FINAL TEST (with best playbook)")
                logger.info(f"{'='*60}\n")
                final_test_results = self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=self.best_playbook,
                    config=config,
                    log_dir=log_dir,
                    save_path=save_path,
                    prefix="final"
                )
                results['final_test_results'] = final_test_results
                logger.info(f"Final Test Accuracy: {final_test_results['accuracy']:.3f}\n")

        elif mode == 'online':
            # ONLINE MODE WORKFLOW
            # 1. Run initial test
            logger.info(f"\n{'='*60}")
            logger.info(f"INITIAL TEST (before training)")
            logger.info(f"{'='*60}\n")
            initial_test_results = self._run_test(
                test_samples=test_samples,
                data_processor=data_processor,
                playbook=self.playbook,
                config=config,
                log_dir=log_dir,
                save_path=save_path,
                prefix="initial"
            )
            results['initial_test_results'] = initial_test_results
            logger.info(f"Initial Test Accuracy: {initial_test_results['accuracy']:.3f}\n")

            # 2. Run online training and testing
            logger.info(f"\n{'='*60}")
            logger.info(f"STARTING ONLINE TRAIN AND TEST")
            logger.info(f"{'='*60}\n")
            online_results = self._online_train_and_test(
                test_samples=test_samples,
                data_processor=data_processor,
                config=config,
                save_path=save_path,
                usage_log_path=usage_log_path,
                playbook_dir=playbook_dir,
                log_dir=log_dir,
                start_window=start_window,
                start_global_step=start_global_step,
                prior_online_results=prior_online_results,
            )
            results['online_test_results'] = online_results
            if online_results.get("latency_stats"):
                results['latency_stats'] = online_results["latency_stats"]
        
        else:  # eval_only
            # EVAL ONLY MODE WORKFLOW
            logger.info(f"\n{'='*60}")
            logger.info(f"RUNNING TEST")
            logger.info(f"{'='*60}\n")
            test_results = self._run_test(
                test_samples=test_samples,
                data_processor=data_processor,
                playbook=self.playbook,
                config=config,
                log_dir=log_dir,
                save_path=save_path,
                prefix="test"
            )
            results['test_results'] = test_results
        
        # Save consolidated results
        final_results_path = os.path.join(save_path, "final_results.json")
        with open(final_results_path, "w") as f:
            json.dump(results, f, indent=2)

        tracker = ExperimentTracker(save_path, os.path.basename(save_path))
        tracker.log_params({"task_name": task_name, "mode": mode, **config})
        tracker.log_metrics(results)
        tracker.finish()
        
        # Print final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"RUN COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Mode: {mode.upper().replace('_', ' ')}")
        if mode == 'offline':
            logger.info(f"Best Validation Accuracy: {results['training_results']['best_validation_accuracy']:.3f}")
            if test_samples:
                logger.info(f"Initial Test Accuracy: {results['initial_test_results']['accuracy']:.3f}")
                logger.info(f"Final Test Accuracy: {results['final_test_results']['accuracy']:.3f}")
        elif mode == 'online':
            logger.info(f"Initial Test Accuracy: {results['initial_test_results']['accuracy']:.3f}")
            logger.info(f"Final Test Accuracy: {results['online_test_results']['accuracy']:.3f}")
        else:  # eval_only
            logger.info(f"Test Accuracy: {results['test_results']['accuracy']:.3f}")
        logger.info(f"Results saved to: {save_path}")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def _run_test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        playbook: str,
        config: Dict[str, Any],
        log_dir: str,
        save_path: str,
        prefix: str = "test"
    ) -> Dict[str, Any]:
        """
        Run testing
        
        Args:
            test_samples: List of test samples
            data_processor: Data processor instance for the task
            playbook: Playbook to use for testing
            config: Configuration dictionary
            log_dir: Directory for detailed logs
            save_path: Path to save results
            prefix: Prefix for saved files (e.g., 'initial', 'final', 'test')
            
        Returns:
            Dictionary with test results
        """
        config_params = self._extract_config_params(config)
        use_json_mode = config_params['use_json_mode']
        test_workers = config_params['test_workers']
        
        test_results, test_error_log = evaluate_test_set(
            data_processor,
            self.generator,
            playbook,
            test_samples,
            self.max_tokens,
            log_dir,
            max_workers=test_workers,
            use_json_mode=use_json_mode
        )

        # Save test results
        test_results_path = os.path.join(save_path, f"{prefix}_test_results.json")
        with open(test_results_path, "w") as f:
            json.dump({
                "test_results": test_results,
                "error_log": test_error_log,
            }, f, indent=2)
        
        return test_results
    
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
    

    
    def test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        playbook,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run testing with the playbook (backward compatibility wrapper).
        
        Args:
            test_samples: List of test samples
            data_processor: Data processor instance for the task
            playbook: Playbook to be used for generator
            config: Configuration dictionary
            
        Returns:
            Dictionary with test results
        """
        # Temporarily set the playbook
        old_playbook = self.playbook
        self.playbook = playbook
        
        # Use the run method
        results = self.run(
            mode='eval_only',
            test_samples=test_samples,
            data_processor=data_processor,
            config=config
        )
        
        # Restore old playbook
        self.playbook = old_playbook
        
        # Return in the old format for backward compatibility
        return {
            "test_results": results['test_results'],
            "error_log": results.get('test_error_log', {}),
            "playbook": playbook
        }
    
