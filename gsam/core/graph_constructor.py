"""
GSAM Graph Constructor

Converts Curator deltas (natural language insights) into typed graph
operations: node additions, edge additions, and attribute updates.

Implements Algorithm 1 from the paper (Graph Construction Pipeline).
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple

from ..graph_memory import KnowledgeGraph, NodeType, EdgeType
from .graph_operation import GraphOperation  # noqa: F401  (re-exported)
from .reflection_extraction import ReflectionExtractionMixin
from ..prompts.graph_constructor import GRAPH_CONSTRUCTOR_PROMPT
from playbook_utils import extract_json_from_text
from llm import timed_llm_call

from log_config import get_logger

logger = get_logger("graph_constructor")


class GraphConstructor(ReflectionExtractionMixin):
    """
    Converts Curator output into graph operations.

    Uses an LLM to extract entities and relationships from natural
    language deltas, then applies them to the knowledge graph with
    de-duplication and ontology anchoring.
    """

    def __init__(
        self,
        api_client,
        api_provider: str,
        model: str,
        graph: KnowledgeGraph,
        entity_name_map: Dict[str, str],
        max_tokens: int = 4096,
        merge_threshold: float = 0.9,
        no_failure_cascades: bool = False,
    ):
        """
        Initialize the GraphConstructor.

        Args:
            api_client: OpenAI-compatible client.
            api_provider: API provider name.
            model: LLM model name.
            graph: The KnowledgeGraph to update.
            entity_name_map: Dict mapping entity names to concept node IDs.
            max_tokens: Max tokens for LLM calls.
            merge_threshold: Cosine similarity threshold for node merging.
            no_failure_cascades: Ablation flag to skip anti-pattern creation.
        """
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.graph = graph
        self.entity_name_map = entity_name_map
        self.max_tokens = max_tokens
        self.merge_threshold = merge_threshold
        self.no_failure_cascades = no_failure_cascades

    def process_curator_output(
        self,
        curator_output: str,
        reflection_content: str,
        task_context: str,
        call_id: str = "gc",
        log_dir: Optional[str] = None,
    ) -> List[GraphOperation]:
        """
        Process curator output and apply graph operations.

        This is the main entry point. It:
        1. Parses curator structured operations (ADD_STRATEGY, etc.)
        2. Converts them into graph operations
        3. Applies operations with de-duplication

        Args:
            curator_output: Raw curator response string.
            reflection_content: The reflector output for additional context.
            task_context: The question/context for this task.
            call_id: Unique call identifier for logging.
            log_dir: Directory for logging.

        Returns:
            List of GraphOperation objects that were applied.
        """
        operations = []

        # Parse the curator output
        curator_json = extract_json_from_text(curator_output)
        if not curator_json:
            return operations

        curator_ops = curator_json.get("operations", [])

        for op in curator_ops:
            op_type = op.get("type", "")

            if op_type == "ADD_STRATEGY":
                ops = self._handle_add_strategy(op)
                operations.extend(ops)
            elif op_type in ("ADD_ANTIPATTERN", "ADD_ANTI_PATTERN"):
                if not self.no_failure_cascades:
                    ops = self._handle_add_antipattern(op)
                    operations.extend(ops)
            elif op_type == "ADD_CONFUSION":
                ops = self._handle_add_confusion(op)
                operations.extend(ops)
            elif op_type == "ADD_PART_OF":
                ops = self._handle_add_part_of(op)
                operations.extend(ops)
            elif op_type == "ADD_CONFLICTS_WITH":
                ops = self._handle_add_conflicts_with(op)
                operations.extend(ops)
            elif op_type == "ADD":
                # Fallback for ACE-style ADD operations
                ops = self._handle_legacy_add(op)
                operations.extend(ops)

        # Also try to extract from reflection if curator output was sparse
        if not operations and reflection_content and reflection_content != "(empty)":
            operations = self._extract_from_reflection(
                reflection_content, call_id, log_dir
            )

        # Apply all operations to graph
        self._apply_operations(operations)

        return operations

    def _handle_add_strategy(self, op: Dict) -> List[GraphOperation]:
        """Convert an ADD_STRATEGY curator operation to graph operations."""
        content = op.get("content", "")
        concepts = op.get("concepts", [])
        fixes_ap = op.get("fixes_antipattern", "")

        if not content:
            return []

        ops = []

        # Check for duplicate strategy
        existing = self.graph.find_similar_node(
            content, NodeType.STRATEGY, self.merge_threshold
        )

        if existing:
            # Merge into existing
            ops.append(GraphOperation(
                op_type="UPDATE_ATTR",
                node_id=existing,
                attrs={"helpful_count": 1},
            ))
            strategy_id = existing
        else:
            # Create new strategy node
            strategy_id = self.graph._generate_id(NodeType.STRATEGY)
            ops.append(GraphOperation(
                op_type="ADD_NODE",
                node_type=NodeType.STRATEGY.value,
                content=content,
                node_id=strategy_id,
            ))

        # Link to concepts via applies_to
        for concept_name in concepts:
            concept_nid = self._resolve_concept(concept_name)
            if concept_nid:
                ops.append(GraphOperation(
                    op_type="ADD_EDGE",
                    source_id=strategy_id,
                    target_id=concept_nid,
                    edge_type=EdgeType.APPLIES_TO.value,
                ))

        # Link to anti-pattern via fixes
        if fixes_ap:
            ap_node = self.graph.find_similar_node(
                fixes_ap, NodeType.ANTI_PATTERN, 0.8
            )
            if ap_node:
                ops.append(GraphOperation(
                    op_type="ADD_EDGE",
                    source_id=strategy_id,
                    target_id=ap_node,
                    edge_type=EdgeType.FIXES.value,
                ))

        return ops

    def _handle_add_antipattern(self, op: Dict) -> List[GraphOperation]:
        """Convert an ADD_ANTIPATTERN curator operation to graph operations."""
        content = op.get("content", "")
        root_cause = op.get("root_cause", "")
        severity = op.get("severity", "medium")
        concepts = op.get("concepts", [])
        cascading = op.get("cascading_formulas", [])

        if not content:
            return []

        ops = []

        # Check for duplicate
        existing = self.graph.find_similar_node(
            content, NodeType.ANTI_PATTERN, self.merge_threshold
        )

        if existing:
            ops.append(GraphOperation(
                op_type="UPDATE_ATTR",
                node_id=existing,
                attrs={"harmful_count": 1},
            ))
            ap_id = existing
        else:
            ap_id = self.graph._generate_id(NodeType.ANTI_PATTERN)
            ops.append(GraphOperation(
                op_type="ADD_NODE",
                node_type=NodeType.ANTI_PATTERN.value,
                content=content,
                node_id=ap_id,
                attrs={
                    "root_cause": root_cause,
                    "severity": severity,
                },
            ))

        # Link to concepts via fails_for
        for concept_name in concepts:
            concept_nid = self._resolve_concept(concept_name)
            if concept_nid:
                ops.append(GraphOperation(
                    op_type="ADD_EDGE",
                    source_id=ap_id,
                    target_id=concept_nid,
                    edge_type=EdgeType.FAILS_FOR.value,
                ))

        # Link to formulas via fails_for (failure cascade)
        for formula_name in cascading:
            formula_nodes = [
                nid for nid, data in self.graph.graph.nodes(data=True)
                if data.get("type") == NodeType.FORMULA.value
                and formula_name.lower() in data.get("name", "").lower()
            ]
            for fid in formula_nodes:
                ops.append(GraphOperation(
                    op_type="ADD_EDGE",
                    source_id=ap_id,
                    target_id=fid,
                    edge_type=EdgeType.FAILS_FOR.value,
                ))

        return ops

    def _handle_add_confusion(self, op: Dict) -> List[GraphOperation]:
        """Convert an ADD_CONFUSION curator operation to graph operations."""
        content = op.get("content", "")
        pair = op.get("concept_pair", [])
        criteria = op.get("distinguishing_criteria", "")

        if not content or len(pair) < 2:
            return []

        ops = []

        # Create confusion node
        conf_id = self.graph._generate_id(NodeType.CONFUSION)
        ops.append(GraphOperation(
            op_type="ADD_NODE",
            node_type=NodeType.CONFUSION.value,
            content=content,
            node_id=conf_id,
            attrs={"distinguishing_criteria": criteria},
        ))

        # Link concepts with confused_with edges
        c1 = self._resolve_concept(pair[0])
        c2 = self._resolve_concept(pair[1])
        if c1 and c2:
            ops.append(GraphOperation(
                op_type="ADD_EDGE",
                source_id=c1,
                target_id=c2,
                edge_type=EdgeType.CONFUSED_WITH.value,
            ))
            ops.append(GraphOperation(
                op_type="ADD_EDGE",
                source_id=c2,
                target_id=c1,
                edge_type=EdgeType.CONFUSED_WITH.value,
            ))

        return ops



    def _handle_legacy_add(self, op: Dict) -> List[GraphOperation]:
        """Handle ACE-style ADD operations (backward compatibility)."""
        content = op.get("content", "")
        if not content:
            return []

        # Treat as a strategy by default
        existing = self.graph.find_similar_node(
            content, NodeType.STRATEGY, self.merge_threshold
        )

        ops = []
        if existing:
            ops.append(GraphOperation(
                op_type="UPDATE_ATTR",
                node_id=existing,
                attrs={"helpful_count": 1},
            ))
        else:
            ops.append(GraphOperation(
                op_type="ADD_NODE",
                node_type=NodeType.STRATEGY.value,
                content=content,
            ))

        return ops




    def _resolve_concept(self, concept_name: str) -> Optional[str]:
        """Resolve a concept name to its node ID in the graph."""
        # Exact match
        if concept_name in self.entity_name_map:
            return self.entity_name_map[concept_name]

        # Case-insensitive match
        lower_name = concept_name.lower()
        for name, nid in self.entity_name_map.items():
            if name.lower() == lower_name:
                return nid

        # Partial match (concept name contains the query)
        for name, nid in self.entity_name_map.items():
            if lower_name in name.lower() or name.lower() in lower_name:
                return nid

        return None

    def _apply_operations(self, operations: List[GraphOperation]) -> None:
        """Apply a list of graph operations to the knowledge graph."""
        for op in operations:
            try:
                if op.op_type == "ADD_NODE":
                    if op.node_id and op.node_id not in self.graph.graph:
                        self.graph.add_node(
                            node_type=NodeType(op.node_type) if op.node_type else NodeType.STRATEGY,
                            content=op.content or "",
                            node_id=op.node_id,
                            **op.attrs,
                        )
                elif op.op_type == "ADD_EDGE":
                    if (op.source_id and op.target_id
                            and op.source_id in self.graph.graph
                            and op.target_id in self.graph.graph):
                        # Avoid duplicate edges
                        if not self.graph.graph.has_edge(op.source_id, op.target_id):
                            try:
                                self.graph.add_edge(
                                    op.source_id,
                                    op.target_id,
                                    EdgeType(op.edge_type) if op.edge_type else EdgeType.APPLIES_TO,
                                )
                            except ValueError:
                                # Unknown edge type, use as string
                                self.graph.graph.add_edge(
                                    op.source_id, op.target_id,
                                    type=op.edge_type or "related_to",
                                    weight=1.0,
                                )
                elif op.op_type == "UPDATE_ATTR":
                    if op.node_id and op.node_id in self.graph.graph:
                        node = self.graph.graph.nodes[op.node_id]
                        for k, v in op.attrs.items():
                            if k in ("helpful_count", "harmful_count", "use_count"):
                                node[k] = node.get(k, 0) + v
                            else:
                                node[k] = v
            except Exception as e:
                logger.warning(f"  Warning: Failed to apply graph op {op.op_type}: {e}")

    def update_node_tags(self, node_tags: List[Dict[str, str]]) -> None:
        """
        Update node helpful/harmful counts based on reflector tags.

        Args:
            node_tags: List of {"id": "S:0001", "tag": "helpful"} dicts.
        """
        for tag_info in node_tags:
            nid = tag_info.get("id", "")
            tag = tag_info.get("tag", "neutral")

            if nid not in self.graph.graph:
                continue

            node = self.graph.graph.nodes[nid]
            if tag == "helpful":
                node["helpful_count"] = node.get("helpful_count", 0) + 1
            elif tag == "harmful":
                node["harmful_count"] = node.get("harmful_count", 0) + 1
            node["use_count"] = node.get("use_count", 0) + 1
            node["last_used"] = __import__("time").time()
