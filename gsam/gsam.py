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

from log_config import get_logger
from experiment_tracker import ExperimentTracker
from resume_utils import find_latest_checkpoint, write_progress, default_resume_state, apply_progress_files

logger = get_logger("gsam")


from .sample_training import SampleTrainingMixin
from .training_loops import TrainingLoopsMixin
from .core.agents import GSAMGenerator, GSAMReflector, GSAMCurator, _RetrievalAwareGenerator


class GSAM(TrainingLoopsMixin, SampleTrainingMixin):
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
            logger.info("Ontology initialization skipped (no_ontology=True or no taxonomy_path)")

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

        logger.info(f"GSAM initialized: {self.knowledge_graph}")

    # ------------------------------------------------------------------
    # Resume helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_latest_graph_checkpoint(graph_dir: str) -> Optional[str]:
        """Return path to the graph checkpoint with the highest step number."""
        return find_latest_checkpoint(graph_dir, '.json')

    def _save_progress(self, save_path: str, **kwargs) -> None:
        """Write progress.json so runs can be resumed after interruption."""
        write_progress(save_path, **kwargs)

    def _load_resume_state(self, resume_path: str) -> Dict[str, Any]:
        """
        Load graph checkpoint and progress from a partial run.
        Mutates self.knowledge_graph in-place from the latest checkpoint.
        Returns a dict with start_epoch, start_step, start_window,
        start_global_step, prior_online_results.
        """
        state = default_resume_state()
        graph_dir = os.path.join(resume_path, "graph_checkpoints")
        latest_ckpt = self._find_latest_graph_checkpoint(graph_dir)
        if latest_ckpt:
            loaded = KnowledgeGraph.load(latest_ckpt)
            # Replace graph internals in-place so existing references
            # (GraphConstructor, GraphRetriever) stay valid.
            self.knowledge_graph.graph = loaded.graph
            self.knowledge_graph.embeddings = loaded.embeddings
            self.knowledge_graph._next_ids = loaded._next_ids
            self.knowledge_graph.tasks_processed = loaded.tasks_processed
            self.knowledge_graph.created_at = loaded.created_at
            logger.info(f"Loaded graph checkpoint: {latest_ckpt}")
            logger.info(f"  Resumed graph: {self.knowledge_graph}")

        return apply_progress_files(resume_path, state, logger)

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

        # Determine save_path: resume into existing dir OR create a new timestamped one
        resume_cfg = config.get('resume_path')
        start_epoch, start_step, start_window, start_global_step = 1, 0, 0, 0
        prior_online_results = None

        if resume_cfg and os.path.isdir(resume_cfg):
            save_path = resume_cfg
            logger.info(f"\nRESUMING run from: {resume_cfg}")
            state = self._load_resume_state(resume_cfg)
            start_epoch = state["start_epoch"]
            start_step = state["start_step"]
            start_window = state["start_window"]
            start_global_step = state["start_global_step"]
            prior_online_results = state["prior_online_results"]
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_folder = f"gsam_run_{timestamp}_{task_name}_{mode}"
            save_path = os.path.join(save_dir, run_folder)
            os.makedirs(save_path, exist_ok=True)

        log_dir = os.path.join(save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        graph_dir = os.path.join(save_path, "graph_checkpoints")
        os.makedirs(graph_dir, exist_ok=True)

        # Save config (overwrite so resume picks up any changed params)
        with open(os.path.join(save_path, "run_config.json"), "w") as f:
            json.dump({"task_name": task_name, "mode": mode, "config": config}, f, indent=2)

        if not resume_cfg:
            # Save initial graph only on a fresh run
            self.knowledge_graph.save(os.path.join(graph_dir, "graph_step_0.json"))

        logger.info(f"\n{'='*60}")
        logger.info(f"GSAM SYSTEM - {mode.upper()} MODE")
        logger.info(f"{'='*60}")
        logger.info(f"Task: {task_name}")
        logger.info(f"Graph: {self.knowledge_graph}")
        logger.info(f"{'='*60}\n")

        results = {}

        if mode == 'offline':
            if test_samples:
                initial = self._run_test(test_samples, data_processor, config, log_dir, save_path, "initial")
                results['initial_test_results'] = initial
                logger.info(f"Initial Test Accuracy: {initial['accuracy']:.3f}\n")

            training_results = self._offline_train(
                train_samples, val_samples, data_processor, config,
                save_path, log_dir, graph_dir,
                start_epoch=start_epoch, start_step=start_step,
            )
            results['training_results'] = training_results
            if training_results.get("latency_stats"):
                results['latency_stats'] = training_results["latency_stats"]

            if test_samples:
                final = self._run_test(test_samples, data_processor, config, log_dir, save_path, "final")
                results['final_test_results'] = final
                logger.info(f"Final Test Accuracy: {final['accuracy']:.3f}\n")

        elif mode == 'online':
            initial = self._run_test(test_samples, data_processor, config, log_dir, save_path, "initial")
            results['initial_test_results'] = initial
            logger.info(f"Initial Test Accuracy: {initial['accuracy']:.3f}\n")

            online_results = self._online_train_and_test(
                test_samples, data_processor, config,
                save_path, log_dir, graph_dir,
                start_window=start_window,
                start_global_step=start_global_step,
                prior_online_results=prior_online_results,
            )
            results['online_test_results'] = online_results
            if online_results.get("latency_stats"):
                results['latency_stats'] = online_results["latency_stats"]

        else:  # eval_only
            test_results = self._run_test(test_samples, data_processor, config, log_dir, save_path, "test")
            results['test_results'] = test_results

        # Save final graph and results
        self.knowledge_graph.save(os.path.join(graph_dir, "graph_final.json"))
        self._save_tracking_logs(save_path)

        with open(os.path.join(save_path, "final_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        tracker = ExperimentTracker(save_path, os.path.basename(save_path))
        tracker.log_params({"task_name": task_name, "mode": mode, **config})
        tracker.log_metrics(results)
        tracker.finish()

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"GSAM RUN COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Final Graph: {self.knowledge_graph}")
        logger.info(f"Results saved to: {save_path}")
        logger.info(f"{'='*60}\n")

        return results

    def _run_test(self, test_samples, data_processor, config, log_dir, save_path, prefix):
        """Run test evaluation using current graph with per-query retrieval."""
        config_params = self._extract_config_params(config)

        # Use per-query graph retrieval at test time so GSAM's retrieval
        # architecture is exercised during evaluation (not just training).
        retrieval_gen = _RetrievalAwareGenerator(self.generator, self.graph_retriever)

        test_results, test_error_log = evaluate_test_set(
            data_processor,
            retrieval_gen,
            "",  # playbook unused — retrieval_gen ignores it
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
        """Get graph serialization for test-time evaluation, capped to prevent context overflow."""
        strategies = self.knowledge_graph.get_nodes_by_type(NodeType.STRATEGY)
        antipatterns = self.knowledge_graph.get_nodes_by_type(NodeType.ANTI_PATTERN)

        if not strategies and not antipatterns:
            return "(No learned knowledge in graph yet)"

        # Rank by helpful_count * confidence; keep top nodes to stay within token budget.
        def _score(nid):
            data = self.knowledge_graph.graph.nodes.get(nid, {})
            return data.get("helpful_count", 0) * data.get("confidence", 1.0) + data.get("helpful_count", 0)

        strategies = sorted(strategies, key=_score, reverse=True)[:20]
        antipatterns = sorted(antipatterns, key=_score, reverse=True)[:10]

        all_ids = strategies + antipatterns
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
