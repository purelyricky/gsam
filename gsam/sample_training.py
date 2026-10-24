"""Single-sample generate / reflect / curate step for the GSAM orchestrator, split out of gsam.py as a mixin."""

import os
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .graph_memory import KnowledgeGraph, NodeType, EdgeType
from .core.agents import GSAMGenerator, GSAMReflector, GSAMCurator, _RetrievalAwareGenerator
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


class SampleTrainingMixin:
    """Per-sample training step shared by the offline and online loops."""

    def _train_single_sample(
        self,
        task_dict: Dict,
        data_processor,
        step_id: str,
        step: int,
        log_dir: str,
        config_params: Dict,
        total_samples: int,
    ) -> Tuple[str, str, Dict, Dict]:
        """Train on a single sample using graph-structured memory."""
        max_num_rounds = config_params['max_num_rounds']
        curator_frequency = config_params['curator_frequency']
        token_budget = config_params['token_budget']
        use_json_mode = config_params['use_json_mode']
        no_ground_truth = config_params['no_ground_truth']

        question = task_dict.get("question", "")
        context = task_dict.get("context", "")
        target = task_dict.get("target", "")

        # Per-sample latency accumulators
        _generator_time = 0.0
        _reflector_time = 0.0
        _curator_time = 0.0
        _graph_update_time = 0.0

        # STEP 1: Graph Retrieval
        start_retrieval = time.time()
        graph_context, retrieved_ids = self.graph_retriever.retrieve(
            query=question, context=context
        )
        retrieval_time = time.time() - start_retrieval

        # STEP 2: Initial Generation
        _t0 = time.time()
        gen_response, node_ids, call_info = self.generator.generate(
            question=question,
            graph_context=graph_context,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_gen_initial",
            log_dir=log_dir,
        )
        _generator_time += time.time() - _t0

        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)
        pre_train_answer = final_answer

        # Log retrieval precision
        referenced_ids = set(node_ids) & set(retrieved_ids)
        retrieval_precision = (
            len(referenced_ids) / len(retrieved_ids) if retrieved_ids else 0.0
        )
        self.retrieval_logs.append({
            "step": step,
            "retrieved_count": len(retrieved_ids),
            "referenced_count": len(referenced_ids),
            "precision": retrieval_precision,
            "retrieval_time_s": retrieval_time,
        })

        logger.info(f"Correct: {is_correct} | Retrieved: {len(retrieved_ids)} nodes | Precision: {retrieval_precision:.2f}")

        tracking_dict = {
            "pre_train_result": {
                "final_answer": final_answer,
                "is_correct": is_correct,
                "retrieved_nodes": len(retrieved_ids),
                "retrieval_precision": retrieval_precision,
            }
        }

        reflection_content = "(empty)"

        # STEP 3: Reflection
        nodes_used_str = self._format_nodes_for_reflector(node_ids)

        if not is_correct:
            for round_num in range(max_num_rounds):
                logger.info(f"Reflection round {round_num + 1}/{max_num_rounds}")

                _t0 = time.time()
                reflection_content, node_tags, _ = self.reflector.reflect(
                    question=question,
                    reasoning_trace=gen_response,
                    predicted_answer=final_answer,
                    ground_truth=target if not no_ground_truth else None,
                    environment_feedback="Predicted answer does not match ground truth",
                    nodes_used=nodes_used_str,
                    use_ground_truth=not no_ground_truth,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_round_{round_num}",
                    log_dir=log_dir,
                )
                _reflector_time += time.time() - _t0

                # Update node counts
                if node_tags:
                    self.graph_constructor.update_node_tags(node_tags)

                # Track error for RFR
                meta = self.reflector.extract_reflection_metadata(reflection_content)
                self.error_history.append({
                    "step": step,
                    "is_correct": False,
                    "concepts_involved": meta.get("concepts_involved", []),
                    "confusion_pairs": meta.get("confusion_pairs", []),
                    "error_severity": meta.get("error_severity", "medium"),
                })

                # Re-retrieve after node tag updates so the generator sees
                # updated helpful/harmful counts in this round's context.
                _t0 = time.time()
                graph_context, retrieved_ids = self.graph_retriever.retrieve(
                    query=question, context=context
                )
                retrieval_time += time.time() - _t0

                # Regenerate with reflection and fresh context
                _t0 = time.time()
                gen_response, node_ids, _ = self.generator.generate(
                    question=question,
                    graph_context=graph_context,
                    context=context,
                    reflection=reflection_content,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_post_reflect_{round_num}",
                    log_dir=log_dir,
                )
                _generator_time += time.time() - _t0

                final_answer = extract_answer(gen_response)
                if data_processor.answer_is_correct(final_answer, target):
                    logger.info(f"Corrected after reflection round {round_num + 1}!")
                    is_correct = True
                    break
        else:
            # Correct answer - still reflect to tag helpful nodes
            _t0 = time.time()
            reflection_content, node_tags, _ = self.reflector.reflect(
                question=question,
                reasoning_trace=gen_response,
                predicted_answer=final_answer,
                ground_truth=target if not no_ground_truth else None,
                environment_feedback="Predicted answer matches ground truth",
                nodes_used=nodes_used_str,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=f"{step_id}_reflect_correct",
                log_dir=log_dir,
            )
            _reflector_time += time.time() - _t0
            if node_tags:
                self.graph_constructor.update_node_tags(node_tags)

        # STEP 4: Curator + Graph Constructor
        if step % curator_frequency == 0:
            logger.info(f"\n--- Running Curator + Graph Constructor at step {step} ---")

            graph_stats_str = json.dumps(self.knowledge_graph.stats(), indent=2)
            graph_summary = self._get_graph_summary_for_curator()

            _t0 = time.time()
            curator_response, _ = self.curator.curate(
                graph_stats=graph_stats_str,
                graph_summary=graph_summary,
                recent_reflection=reflection_content,
                question_context=context,
                current_step=step,
                total_samples=total_samples,
                token_budget=token_budget,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=step_id,
                log_dir=log_dir,
            )
            _curator_time += time.time() - _t0

            # Convert curator output to graph operations
            if not curator_response.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE"):
                _t0 = time.time()
                ops = self.graph_constructor.process_curator_output(
                    curator_output=curator_response,
                    reflection_content=reflection_content,
                    task_context=context,
                    call_id=step_id,
                    log_dir=log_dir,
                )
                _graph_update_time += time.time() - _t0
                logger.info(f"  Applied {len(ops)} graph operations")

        # STEP 5: Periodic pruning
        if step % self.prune_frequency == 0 and step > 0:
            pruned = self.knowledge_graph.prune()
            if pruned:
                logger.info(f"  Pruned {pruned} low-utility nodes")

        # Post-curator generation: only re-generate when the curator
        # actually ran and updated the graph, otherwise the context is
        # unchanged and the extra LLM call is wasted.
        if step % curator_frequency == 0:
            _t0 = time.time()
            graph_context_post, _ = self.graph_retriever.retrieve(query=question, context=context)
            retrieval_time += time.time() - _t0
            _t0 = time.time()
            gen_response, _, _ = self.generator.generate(
                question=question,
                graph_context=graph_context_post,
                context=context,
                reflection="(empty)",
                use_json_mode=use_json_mode,
                call_id=f"{step_id}_post_curate",
                log_dir=log_dir,
            )
            _generator_time += time.time() - _t0
            final_answer = extract_answer(gen_response)

        post_train_answer = final_answer
        post_correct = data_processor.answer_is_correct(final_answer, target)

        self.knowledge_graph.tasks_processed += 1

        tracking_dict["post_train_result"] = {
            "final_answer": final_answer,
            "is_correct": post_correct,
            "graph_stats": self.knowledge_graph.stats(),
        }

        latency_dict = {
            "generator_time": _generator_time,
            "reflector_time": _reflector_time,
            "curator_time": _curator_time,
            "retrieval_time": retrieval_time,
            "graph_update_time": _graph_update_time,
        }

        return pre_train_answer, post_train_answer, tracking_dict, latency_dict
