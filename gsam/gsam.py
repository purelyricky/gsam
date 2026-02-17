"""
GSAM (Graph-Structured Adaptive Memory) Orchestrator

Extends ACE's orchestrator by replacing flat bullet storage with
a graph-structured knowledge memory. Inherits the Generator/Reflector/Curator
architecture and adds GraphConstructor and GraphRetriever.
"""

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


class GSAMGenerator:
    """Generator agent adapted for GSAM's graph-structured context."""

    def __init__(self, api_client, api_provider, model, max_tokens=4096):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        question: str,
        graph_context: str = "",
        context: str = "",
        reflection: str = "(empty)",
        use_json_mode: bool = False,
        call_id: str = "gen",
        log_dir: Optional[str] = None,
        # For compatibility with ACE's evaluate_test_set
        playbook: str = "",
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Generate an answer using graph-structured context.

        Args:
            question: The question to answer.
            graph_context: Serialized subgraph from GraphRetriever.
            context: Additional task context.
            reflection: Previous reflection content.
            use_json_mode: Whether to use JSON mode.
            call_id: Unique call identifier.
            log_dir: Logging directory.
            playbook: Unused, for API compatibility with ACE's evaluate_test_set.

        Returns:
            Tuple of (response, node_ids_used, call_info).
        """
        # Use graph_context if provided, otherwise fall back to playbook
        actual_context = graph_context if graph_context else playbook

        prompt = GSAM_GENERATOR_PROMPT.format(
            actual_context, reflection, question, context
        )

        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="generator",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
        )

        node_ids = self._extract_node_ids(response, use_json_mode)

        return response, node_ids, call_info

    def _extract_node_ids(self, response: str, use_json_mode: bool) -> List[str]:
        """Extract node IDs from generator response."""
        node_ids = []

        if use_json_mode:
            try:
                parsed = json.loads(response)
                node_ids = parsed.get("node_ids", [])
            except (json.JSONDecodeError, KeyError):
                node_ids = self._extract_node_ids_regex(response)
        else:
            node_ids = self._extract_node_ids_regex(response)

        return node_ids

    def _extract_node_ids_regex(self, text: str) -> List[str]:
        """Extract node IDs using regex: [S:0001], [A:0005], [C:0023], etc."""
        pattern = r'\[([SACFX]:\d{4})\]'
        return re.findall(pattern, text)


class GSAMReflector:
    """Reflector agent adapted for GSAM's concept-level error analysis."""

    def __init__(self, api_client, api_provider, model, max_tokens=4096):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens

    def reflect(
        self,
        question: str,
        reasoning_trace: str,
        predicted_answer: str,
        ground_truth: Optional[str],
        environment_feedback: str,
        nodes_used: str,
        use_ground_truth: bool = True,
        use_json_mode: bool = False,
        call_id: str = "reflect",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Analyze generator output with concept-level error identification.

        Returns:
            Tuple of (reflection_content, node_tags, call_info).
        """
        if use_ground_truth and ground_truth:
            prompt = GSAM_REFLECTOR_PROMPT.format(
                question, reasoning_trace, predicted_answer,
                ground_truth, environment_feedback, nodes_used
            )
        else:
            prompt = GSAM_REFLECTOR_PROMPT_NO_GT.format(
                question, reasoning_trace, predicted_answer,
                environment_feedback, nodes_used
            )

        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="reflector",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
        )

        node_tags = self._extract_node_tags(response)

        return response, node_tags, call_info

    def _extract_node_tags(self, response: str) -> List[Dict[str, str]]:
        """Extract node tags from reflector response."""
        try:
            parsed = extract_json_from_text(response)
            if parsed:
                return parsed.get("node_tags", parsed.get("bullet_tags", []))
        except Exception:
            pass
        return []

    def extract_reflection_metadata(self, response: str) -> Dict[str, Any]:
        """Extract concept-level metadata from reflection."""
        try:
            parsed = extract_json_from_text(response)
            if parsed:
                return {
                    "concepts_involved": parsed.get("concepts_involved", []),
                    "confusion_pairs": parsed.get("confusion_pairs", []),
                    "cascading_effects": parsed.get("cascading_effects", []),
                    "error_severity": parsed.get("error_severity", "medium"),
                    "key_insight": parsed.get("key_insight", ""),
                }
        except Exception:
            pass
        return {}


class GSAMCurator:
    """Curator agent adapted for GSAM's graph-structured output."""

    def __init__(self, api_client, api_provider, model, max_tokens=4096):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens

    def curate(
        self,
        graph_stats: str,
        graph_summary: str,
        recent_reflection: str,
        question_context: str,
        current_step: int,
        total_samples: int,
        token_budget: int,
        use_ground_truth: bool = True,
        use_json_mode: bool = False,
        call_id: str = "curate",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Curate graph updates based on reflection feedback.

        Returns:
            Tuple of (curator_response, call_info).
        """
        if use_ground_truth:
            prompt = GSAM_CURATOR_PROMPT.format(
                token_budget=token_budget,
                current_step=current_step,
                total_samples=total_samples,
                graph_stats=graph_stats,
                recent_reflection=recent_reflection,
                current_graph_summary=graph_summary,
                question_context=question_context,
            )
        else:
            prompt = GSAM_CURATOR_PROMPT_NO_GT.format(
                token_budget=token_budget,
                current_step=current_step,
                total_samples=total_samples,
                graph_stats=graph_stats,
                recent_reflection=recent_reflection,
                current_graph_summary=graph_summary,
                question_context=question_context,
            )

        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="curator",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
        )

        return response, call_info


class GSAM:
    """
    Main GSAM orchestrator.

    Extends the ACE architecture by replacing flat bullet storage with
    a graph-structured knowledge memory.
    """

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        reflector_model: str,
        curator_model: str,
        max_tokens: int = 4096,
        taxonomy_path: Optional[str] = None,
        formula_data_path: Optional[str] = None,
        merge_threshold: float = 0.9,
        retrieval_depth: int = 2,
        prune_frequency: int = 50,
        # Ablation flags
        no_ontology: bool = False,
        no_failure_cascades: bool = False,
        embedding_only_retrieval: bool = False,
        untyped_edges: bool = False,
    ):
        """
        Initialize the GSAM system.

        Args:
            api_provider: API provider name (sambanova, together, openai).
            generator_model: Model for generator.
            reflector_model: Model for reflector.
            curator_model: Model for curator / graph constructor.
            max_tokens: Max tokens for LLM calls.
            taxonomy_path: Path to xbrl_taxonomy.json.
            formula_data_path: Path to formula training data.
            merge_threshold: Cosine similarity for node dedup.
            retrieval_depth: BFS depth for graph retrieval.
            prune_frequency: Prune every N steps.
            no_ontology: Ablation: skip ontology initialization.
            no_failure_cascades: Ablation: skip anti-pattern creation.
            embedding_only_retrieval: Ablation: embedding-only retrieval.
            untyped_edges: Ablation: all edges generic.
        """
        # Initialize API clients
        gen_client, ref_client, cur_client = initialize_clients(api_provider)

        # Initialize agents
        self.generator = GSAMGenerator(gen_client, api_provider, generator_model, max_tokens)
        self.reflector = GSAMReflector(ref_client, api_provider, reflector_model, max_tokens)
        self.curator = GSAMCurator(cur_client, api_provider, curator_model, max_tokens)

        self.max_tokens = max_tokens
        self.prune_frequency = prune_frequency
        self.merge_threshold = merge_threshold

        # Initialize Knowledge Graph
        self.knowledge_graph = KnowledgeGraph()
        if untyped_edges:
            self.knowledge_graph.use_typed_edges = False

        # Initialize ontology
        self.entity_name_map = {}
        if not no_ontology and taxonomy_path:
            self.entity_name_map = initialize_ontology(
                self.knowledge_graph,
                taxonomy_path,
                formula_data_path,
            )
        else:
            print("Ontology initialization skipped (no_ontology=True or no taxonomy_path)")

        # Initialize Graph Constructor
        self.graph_constructor = GraphConstructor(
            api_client=cur_client,
            api_provider=api_provider,
            model=curator_model,
            graph=self.knowledge_graph,
            entity_name_map=self.entity_name_map,
            max_tokens=max_tokens,
            merge_threshold=merge_threshold,
            no_failure_cascades=no_failure_cascades,
        )

        # Initialize Graph Retriever
        self.graph_retriever = GraphRetriever(
            graph=self.knowledge_graph,
            retrieval_depth=retrieval_depth,
            embedding_only=embedding_only_retrieval,
        )

        # Tracking
        self.error_history: List[Dict] = []
        self.retrieval_logs: List[Dict] = []

        print(f"GSAM initialized: {self.knowledge_graph}")

    def run(
        self,
        mode: str,
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
        test_samples: Optional[List[Dict[str, Any]]] = None,
        data_processor=None,
        config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint for GSAM. Mirrors ACE's run() interface.
        """
        if mode not in ['offline', 'online', 'eval_only']:
            raise ValueError(f"Invalid mode: {mode}")

        config = config or {}
        config_params = self._extract_config_params(config)
        save_dir = config_params['save_dir']
        task_name = config_params['task_name']

        # Setup paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"gsam_run_{timestamp}_{task_name}_{mode}"
        save_path = os.path.join(save_dir, run_folder)
        os.makedirs(save_path, exist_ok=True)
        log_dir = os.path.join(save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        graph_dir = os.path.join(save_path, "graph_checkpoints")
        os.makedirs(graph_dir, exist_ok=True)

        # Save config
        with open(os.path.join(save_path, "run_config.json"), "w") as f:
            json.dump({"task_name": task_name, "mode": mode, "config": config}, f, indent=2)

        # Save initial graph
        self.knowledge_graph.save(os.path.join(graph_dir, "graph_step_0.json"))

        print(f"\n{'='*60}")
        print(f"GSAM SYSTEM - {mode.upper()} MODE")
        print(f"{'='*60}")
        print(f"Task: {task_name}")
        print(f"Graph: {self.knowledge_graph}")
        print(f"{'='*60}\n")

        results = {}

        if mode == 'offline':
            if test_samples:
                initial = self._run_test(test_samples, data_processor, config, log_dir, save_path, "initial")
                results['initial_test_results'] = initial
                print(f"Initial Test Accuracy: {initial['accuracy']:.3f}\n")

            training_results = self._offline_train(
                train_samples, val_samples, data_processor, config,
                save_path, log_dir, graph_dir,
            )
            results['training_results'] = training_results

            if test_samples:
                final = self._run_test(test_samples, data_processor, config, log_dir, save_path, "final")
                results['final_test_results'] = final
                print(f"Final Test Accuracy: {final['accuracy']:.3f}\n")

        elif mode == 'online':
            initial = self._run_test(test_samples, data_processor, config, log_dir, save_path, "initial")
            results['initial_test_results'] = initial
            print(f"Initial Test Accuracy: {initial['accuracy']:.3f}\n")

            online_results = self._online_train_and_test(
                test_samples, data_processor, config,
                save_path, log_dir, graph_dir,
            )
            results['online_test_results'] = online_results

        else:  # eval_only
            test_results = self._run_test(test_samples, data_processor, config, log_dir, save_path, "test")
            results['test_results'] = test_results

        # Save final graph and results
        self.knowledge_graph.save(os.path.join(graph_dir, "graph_final.json"))
        self._save_tracking_logs(save_path)

        with open(os.path.join(save_path, "final_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print(f"GSAM RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Final Graph: {self.knowledge_graph}")
        print(f"Results saved to: {save_path}")
        print(f"{'='*60}\n")

        return results

    def _run_test(self, test_samples, data_processor, config, log_dir, save_path, prefix):
        """Run test evaluation using current graph."""
        config_params = self._extract_config_params(config)

        # For test evaluation, serialize the full graph context
        # We use the ACE-compatible evaluate_test_set but with our generator
        # that accepts playbook parameter (for compatibility)
        # The generator will use graph retrieval internally
        graph_context = self._get_full_graph_summary()

        test_results, test_error_log = evaluate_test_set(
            data_processor,
            self.generator,
            graph_context,  # passed as "playbook" parameter
            test_samples,
            self.max_tokens,
            log_dir,
            max_workers=config_params['test_workers'],
            use_json_mode=config_params['use_json_mode'],
        )

        test_results_path = os.path.join(save_path, f"{prefix}_test_results.json")
        with open(test_results_path, "w") as f:
            json.dump({"test_results": test_results, "error_log": test_error_log}, f, indent=2)

        return test_results

    def _train_single_sample(
        self,
        task_dict: Dict,
        data_processor,
        step_id: str,
        step: int,
        log_dir: str,
        config_params: Dict,
        total_samples: int,
    ) -> Tuple[str, str, Dict]:
        """Train on a single sample using graph-structured memory."""
        max_num_rounds = config_params['max_num_rounds']
        curator_frequency = config_params['curator_frequency']
        token_budget = config_params['token_budget']
        use_json_mode = config_params['use_json_mode']
        no_ground_truth = config_params['no_ground_truth']

        question = task_dict.get("question", "")
        context = task_dict.get("context", "")
        target = task_dict.get("target", "")

        # STEP 1: Graph Retrieval
        start_retrieval = time.time()
        graph_context, retrieved_ids = self.graph_retriever.retrieve(
            query=question, context=context
        )
        retrieval_time = time.time() - start_retrieval

        # STEP 2: Initial Generation
        gen_response, node_ids, call_info = self.generator.generate(
            question=question,
            graph_context=graph_context,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_gen_initial",
            log_dir=log_dir,
        )

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

        print(f"Correct: {is_correct} | Retrieved: {len(retrieved_ids)} nodes | Precision: {retrieval_precision:.2f}")

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
                print(f"Reflection round {round_num + 1}/{max_num_rounds}")

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

                # Regenerate with reflection
                gen_response, node_ids, _ = self.generator.generate(
                    question=question,
                    graph_context=graph_context,
                    context=context,
                    reflection=reflection_content,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_post_reflect_{round_num}",
                    log_dir=log_dir,
                )

                final_answer = extract_answer(gen_response)
                if data_processor.answer_is_correct(final_answer, target):
                    print(f"Corrected after reflection round {round_num + 1}!")
                    is_correct = True
                    break
        else:
            # Correct answer - still reflect to tag helpful nodes
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
            if node_tags:
                self.graph_constructor.update_node_tags(node_tags)

        # STEP 4: Curator + Graph Constructor
        if step % curator_frequency == 0:
            print(f"\n--- Running Curator + Graph Constructor at step {step} ---")

            graph_stats_str = json.dumps(self.knowledge_graph.stats(), indent=2)
            graph_summary = self._get_graph_summary_for_curator()

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

            # Convert curator output to graph operations
            if not curator_response.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE"):
                ops = self.graph_constructor.process_curator_output(
                    curator_output=curator_response,
                    reflection_content=reflection_content,
                    task_context=context,
                    call_id=step_id,
                    log_dir=log_dir,
                )
                print(f"  Applied {len(ops)} graph operations")

        # STEP 5: Periodic pruning
        if step % self.prune_frequency == 0 and step > 0:
            pruned = self.knowledge_graph.prune()
            if pruned:
                print(f"  Pruned {pruned} low-utility nodes")

        # Post-curator generation: only re-generate when the curator
        # actually ran and updated the graph, otherwise the context is
        # unchanged and the extra LLM call is wasted.
        if step % curator_frequency == 0:
            graph_context_post, _ = self.graph_retriever.retrieve(query=question, context=context)
            gen_response, _, _ = self.generator.generate(
                question=question,
                graph_context=graph_context_post,
                context=context,
                reflection="(empty)",
                use_json_mode=use_json_mode,
                call_id=f"{step_id}_post_curate",
                log_dir=log_dir,
            )
            final_answer = extract_answer(gen_response)

        post_train_answer = final_answer
        post_correct = data_processor.answer_is_correct(final_answer, target)

        self.knowledge_graph.tasks_processed += 1

        tracking_dict["post_train_result"] = {
            "final_answer": final_answer,
            "is_correct": post_correct,
            "graph_stats": self.knowledge_graph.stats(),
        }

        return pre_train_answer, post_train_answer, tracking_dict

    def _offline_train(self, train_samples, val_samples, data_processor, config,
                       save_path, log_dir, graph_dir):
        """Offline training loop."""
        config_params = self._extract_config_params(config)
        num_epochs = config_params['num_epochs']
        eval_steps = config_params['eval_steps']
        save_steps = config_params['save_steps']

        results = []
        pre_post_results = []
        best_accuracy = 0.0

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}\nEPOCH {epoch}/{num_epochs}\n{'='*60}")

            answers_pre, targets_pre = [], []
            answers_post, targets_post = [], []

            for step, task_dict in enumerate(train_samples):
                step += 1
                print(f"\n--- Step {step}/{len(train_samples)} ---")

                target = task_dict.get("target", "")
                pre, post, tracking = self._train_single_sample(
                    task_dict, data_processor, f"train_e_{epoch}_s_{step}",
                    step, log_dir, config_params, len(train_samples),
                )

                answers_pre.append(pre)
                targets_pre.append(target)
                answers_post.append(post)
                targets_post.append(target)

                pre_post_results.append({
                    "epoch": epoch, "step": step, "target": target, **tracking,
                })

                if step % save_steps == 0:
                    self.knowledge_graph.save(
                        os.path.join(graph_dir, f"graph_epoch_{epoch}_step_{step}.json")
                    )

                if step % eval_steps == 0:
                    pre_acc = data_processor.evaluate_accuracy(answers_pre, targets_pre)
                    post_acc = data_processor.evaluate_accuracy(answers_post, targets_post)

                    val_results = {}
                    if val_samples:
                        graph_ctx = self._get_full_graph_summary()
                        val_results, _ = evaluate_test_set(
                            data_processor, self.generator, graph_ctx,
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
                print(f"Epoch {epoch} consolidation: {consolidation_stats}")
                self.knowledge_graph.save(
                    os.path.join(graph_dir, f"graph_epoch_{epoch}_consolidated.json")
                )

        # Save results
        with open(os.path.join(save_path, "train_results.json"), "w") as f:
            json.dump({"best_accuracy": best_accuracy, "results": results}, f, indent=2)
        with open(os.path.join(save_path, "pre_train_post_train_results.json"), "w") as f:
            json.dump(pre_post_results, f, indent=2)

        return {"best_validation_accuracy": best_accuracy}

    def _online_train_and_test(self, test_samples, data_processor, config,
                               save_path, log_dir, graph_dir):
        """Online training and testing loop."""
        config_params = self._extract_config_params(config)
        online_eval_frequency = config.get('online_eval_frequency', 15)
        save_steps = config_params['save_steps']

        correct_count_sample = 0
        total_count = 0
        all_answers = []
        all_targets = []
        all_errors = []
        window_results = []

        num_windows = (len(test_samples) + online_eval_frequency - 1) // online_eval_frequency
        global_step = 0

        for window_idx in range(num_windows):
            start = window_idx * online_eval_frequency
            end = min((window_idx + 1) * online_eval_frequency, len(test_samples))
            window_samples = test_samples[start:end]

            print(f"\n{'='*60}\nWINDOW {window_idx+1}/{num_windows} (samples {start}-{end-1})\n{'='*60}")

            # Test on window
            graph_ctx = self._get_full_graph_summary()
            w_results, w_errors = evaluate_test_set(
                data_processor, self.generator, graph_ctx,
                window_samples, self.max_tokens, log_dir,
                max_workers=config_params['test_workers'],
                use_json_mode=config_params['use_json_mode'],
            )

            w_acc = w_results['accuracy']
            w_correct = w_results['correct']
            w_total = w_results['total']
            correct_count_sample += w_correct
            total_count += w_total
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
            print(f"Window {window_idx+1} accuracy: {w_acc:.3f} | Cumulative: {cum_acc:.3f}")

            # Train on window
            for local_step, task_dict in enumerate(window_samples):
                global_step += 1
                print(f"\n--- Window {window_idx+1}, Step {local_step+1}/{len(window_samples)} (Global {global_step}) ---")

                self._train_single_sample(
                    task_dict, data_processor, f"online_train_s_{global_step}",
                    global_step, log_dir, config_params, len(test_samples),
                )

                if global_step % save_steps == 0:
                    self.knowledge_graph.save(
                        os.path.join(graph_dir, f"graph_step_{global_step}.json")
                    )

        # Compute final accuracy using the proper per-task metric
        if all_answers and all_targets:
            final_acc = data_processor.evaluate_accuracy(all_answers, all_targets)
        else:
            final_acc = 0.0

        # Save results
        with open(os.path.join(save_path, "test_results.json"), "w") as f:
            json.dump({
                "test_accuracy": final_acc,
                "window_results": window_results,
                "errors": all_errors,
            }, f, indent=2)

        print(f"\nFinal Online Test Accuracy: {final_acc:.3f}")

        return {
            "accuracy": final_acc,
            "correct": correct_count_sample,
            "total": total_count,
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _extract_config_params(self, config):
        """Extract config with defaults."""
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
        }

    def _format_nodes_for_reflector(self, node_ids: List[str]) -> str:
        """Format graph nodes used by generator for reflector input."""
        if not node_ids:
            return "(No graph nodes used by generator)"

        lines = []
        for nid in node_ids:
            if nid in self.knowledge_graph.graph:
                data = self.knowledge_graph.graph.nodes[nid]
                ntype = data.get("type", "")
                content = data.get("content", "")
                h = data.get("helpful_count", 0)
                n = data.get("harmful_count", 0)
                lines.append(f"[{nid}] ({ntype}) helpful={h} harmful={n} :: {content}")

        return "\n".join(lines) if lines else "(Referenced nodes not found in graph)"

    def _get_graph_summary_for_curator(self) -> str:
        """Get a compact summary of the graph for curator context."""
        stats = self.knowledge_graph.stats()
        lines = [f"Graph has {stats['total_nodes']} nodes, {stats['total_edges']} edges"]

        # List recent strategies
        strategies = self.knowledge_graph.get_nodes_by_type(NodeType.STRATEGY)
        if strategies:
            lines.append(f"\nRecent Strategies ({len(strategies)} total):")
            for sid in strategies[-10:]:  # Last 10
                data = self.knowledge_graph.graph.nodes[sid]
                lines.append(f"  [{sid}] {data.get('content', '')[:100]}")

        # List recent anti-patterns
        antipatterns = self.knowledge_graph.get_nodes_by_type(NodeType.ANTI_PATTERN)
        if antipatterns:
            lines.append(f"\nRecent Anti-Patterns ({len(antipatterns)} total):")
            for aid in antipatterns[-5:]:
                data = self.knowledge_graph.graph.nodes[aid]
                lines.append(f"  [{aid}] {data.get('content', '')[:100]}")

        return "\n".join(lines)

    def _get_full_graph_summary(self) -> str:
        """Get full graph serialization for test-time evaluation."""
        # Get all strategies and anti-patterns
        strategies = self.knowledge_graph.get_nodes_by_type(NodeType.STRATEGY)
        antipatterns = self.knowledge_graph.get_nodes_by_type(NodeType.ANTI_PATTERN)

        all_ids = strategies + antipatterns
        if not all_ids:
            return "(No learned knowledge in graph yet)"

        subgraph = self.knowledge_graph.get_subgraph(all_ids, depth=1)
        return self.knowledge_graph.serialize_subgraph(subgraph)

    def _save_tracking_logs(self, save_path: str) -> None:
        """Save retrieval logs and error history."""
        if self.retrieval_logs:
            with open(os.path.join(save_path, "retrieval_logs.jsonl"), "w") as f:
                for entry in self.retrieval_logs:
                    f.write(json.dumps(entry) + "\n")

        if self.error_history:
            with open(os.path.join(save_path, "error_tracking.jsonl"), "w") as f:
                for entry in self.error_history:
                    f.write(json.dumps(entry) + "\n")

        # Save graph statistics over time
        with open(os.path.join(save_path, "graph_stats.json"), "w") as f:
            json.dump(self.knowledge_graph.stats(), f, indent=2)
