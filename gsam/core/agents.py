"""
GSAM agents: Generator, Reflector and Curator adapted to graph-structured context,
plus the retrieval-aware generator wrapper used at evaluation time.
"""

import re
from typing import Dict, List, Tuple, Optional, Any

from ..prompts.generator import GSAM_GENERATOR_PROMPT
from ..prompts.reflector import GSAM_REFLECTOR_PROMPT, GSAM_REFLECTOR_PROMPT_NO_GT
from ..prompts.curator import GSAM_CURATOR_PROMPT, GSAM_CURATOR_PROMPT_NO_GT
from .graph_retriever import GraphRetriever

from llm import timed_llm_call
from playbook_utils import extract_json_from_text
from log_config import get_logger

logger = get_logger("agents")


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
        """Extract node IDs from generator response.

        Always tries JSON extraction first (model is prompted for JSON output),
        then falls back to regex for bracket-style references like [S:0001].
        """
        # Try JSON extraction regardless of use_json_mode — the generator prompt
        # asks for a JSON object with a "node_ids" key in all modes.
        try:
            parsed = extract_json_from_text(response)
            if parsed and isinstance(parsed.get("node_ids"), list):
                return [str(nid) for nid in parsed["node_ids"]]
        except Exception:
            pass

        # Fall back to regex: matches [S:0001], [A:0003], etc.
        return self._extract_node_ids_regex(response)

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
        # Truncate large inputs to prevent oversized prompts causing vLLM None responses.
        MAX_SUMMARY_CHARS = 3000
        if len(graph_summary) > MAX_SUMMARY_CHARS:
            graph_summary = graph_summary[:MAX_SUMMARY_CHARS] + "\n...[truncated for brevity]"

        MAX_REFLECTION_CHARS = 2000
        if len(recent_reflection) > MAX_REFLECTION_CHARS:
            recent_reflection = recent_reflection[:MAX_REFLECTION_CHARS] + "\n...[truncated]"

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


class _RetrievalAwareGenerator:
    """
    Wraps GSAMGenerator to perform per-query graph retrieval at test/eval time.

    evaluate_test_set() passes context via the `playbook` parameter. This
    wrapper intercepts those calls and replaces the static playbook with a
    targeted subgraph retrieved for the specific (question, context) pair,
    which is what GSAM should be doing at inference time.
    """

    def __init__(self, generator: "GSAMGenerator", retriever: "GraphRetriever"):
        self._gen = generator
        self._ret = retriever

    def generate(self, question: str, playbook: str = "", context: str = "", **kwargs):
        # Ignore the static playbook — retrieve per-query instead.
        graph_ctx, _ = self._ret.retrieve(query=question, context=context)
        # Drop 'playbook' from kwargs if callers passed it both ways.
        kwargs.pop("playbook", None)
        return self._gen.generate(
            question=question,
            graph_context=graph_ctx,
            context=context,
            **kwargs,
        )
