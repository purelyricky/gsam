"""LLM-based extraction of graph operations from reflections, plus structural edge handlers; split out of graph_constructor.py as a mixin."""

from typing import Dict, List, Optional

from ..graph_memory import NodeType, EdgeType
from .graph_operation import GraphOperation
from ..prompts.graph_constructor import GRAPH_CONSTRUCTOR_PROMPT
from playbook_utils import extract_json_from_text
from llm import timed_llm_call
from log_config import get_logger

logger = get_logger("graph_constructor")


class ReflectionExtractionMixin:

    def _extract_from_reflection(
        self,
        reflection: str,
        call_id: str,
        log_dir: Optional[str],
    ) -> List[GraphOperation]:
        """
        Use LLM to extract entities/relationships from reflection text.

        This is a fallback when the curator output doesn't contain
        structured graph operations.
        """
        prompt = GRAPH_CONSTRUCTOR_PROMPT.format(reflection)

        try:
            response, _ = timed_llm_call(
                self.api_client,
                self.api_provider,
                self.model,
                prompt,
                role="graph_constructor",
                call_id=f"{call_id}_gc_extract",
                max_tokens=self.max_tokens,
                log_dir=log_dir,
                use_json_mode=False,
            )

            extracted = extract_json_from_text(response)
            if not extracted:
                return []

            return self._parse_extracted_graph(extracted)

        except Exception as e:
            logger.info(f"GraphConstructor extraction failed: {e}")
            return []

    def _parse_extracted_graph(self, data: Dict) -> List[GraphOperation]:
        """Parse LLM-extracted graph data into operations."""
        ops = []
        node_id_map = {}  # "NodeType:Index" -> actual node_id

        # Process nodes
        for i, node in enumerate(data.get("nodes", [])):
            ntype_str = node.get("type", "Strategy")
            content = node.get("content", "")
            if not content:
                continue

            try:
                ntype = NodeType(ntype_str)
            except ValueError:
                ntype = NodeType.STRATEGY

            # Check de-duplication
            existing = self.graph.find_similar_node(
                content, ntype, self.merge_threshold
            )

            if existing:
                node_id_map[f"{ntype_str}:{i}"] = existing
                ops.append(GraphOperation(
                    op_type="UPDATE_ATTR",
                    node_id=existing,
                    attrs={
                        "helpful_count": 1 if ntype == NodeType.STRATEGY else 0,
                        "harmful_count": 1 if ntype == NodeType.ANTI_PATTERN else 0,
                    },
                ))
            else:
                nid = self.graph._generate_id(ntype)
                node_id_map[f"{ntype_str}:{i}"] = nid
                attrs = {}
                for key in ["root_cause", "severity", "distinguishing_criteria"]:
                    if key in node:
                        attrs[key] = node[key]
                ops.append(GraphOperation(
                    op_type="ADD_NODE",
                    node_type=ntype.value,
                    content=content,
                    node_id=nid,
                    attrs=attrs,
                ))

        # Process edges
        for edge in data.get("edges", []):
            src_ref = edge.get("source", "")
            tgt_ref = edge.get("target", "")
            rel = edge.get("relation", "related_to")

            src_id = self._resolve_edge_ref(src_ref, node_id_map)
            tgt_id = self._resolve_edge_ref(tgt_ref, node_id_map)

            if src_id and tgt_id:
                ops.append(GraphOperation(
                    op_type="ADD_EDGE",
                    source_id=src_id,
                    target_id=tgt_id,
                    edge_type=rel,
                ))

        return ops

    def _resolve_edge_ref(
        self, ref: str, node_id_map: Dict[str, str]
    ) -> Optional[str]:
        """Resolve an edge reference to an actual node ID."""
        # Direct match in newly created nodes
        if ref in node_id_map:
            return node_id_map[ref]

        # Concept:EntityName reference
        if ref.startswith("Concept:"):
            entity_name = ref.split(":", 1)[1]
            return self._resolve_concept(entity_name)

        # Formula:FormulaName reference
        if ref.startswith("Formula:"):
            formula_name = ref.split(":", 1)[1]
            for nid, data in self.graph.graph.nodes(data=True):
                if (data.get("type") == NodeType.FORMULA.value
                        and formula_name.lower() in data.get("name", "").lower()):
                    return nid

        # Try as existing node ID
        if ref in self.graph.graph:
            return ref

        return None

    def _handle_add_part_of(self, op: Dict) -> List[GraphOperation]:
        """Convert an ADD_PART_OF curator operation to a part_of edge.

        Expects:
            component: name of the component concept (e.g. "RevenueFromContractWithCustomer")
            aggregate: name of the aggregate concept (e.g. "Revenue")

        Creates a single directed part_of edge from the component concept node
        to the aggregate concept node.  Both concepts must already exist in the
        graph (they are XBRL ontology nodes); if either cannot be resolved the
        operation is silently skipped.
        """
        component_name = op.get("component", "")
        aggregate_name = op.get("aggregate", "")

        if not component_name or not aggregate_name:
            return []

        component_nid = self._resolve_concept(component_name)
        aggregate_nid = self._resolve_concept(aggregate_name)

        if not component_nid or not aggregate_nid:
            return []

        return [GraphOperation(
            op_type="ADD_EDGE",
            source_id=component_nid,
            target_id=aggregate_nid,
            edge_type=EdgeType.PART_OF.value,
        )]

    def _handle_add_conflicts_with(self, op: Dict) -> List[GraphOperation]:
        """Convert an ADD_CONFLICTS_WITH curator operation to conflicts_with edges.

        Expects:
            strategy_a: content of the first strategy (used for similarity lookup)
            strategy_b: content of the second strategy (used for similarity lookup)

        Creates bidirectional conflicts_with edges between the two strategy nodes.
        If either strategy cannot be found in the graph the operation is skipped.
        """
        strategy_a_content = op.get("strategy_a", "")
        strategy_b_content = op.get("strategy_b", "")

        if not strategy_a_content or not strategy_b_content:
            return []

        sid_a = self.graph.find_similar_node(
            strategy_a_content, NodeType.STRATEGY, threshold=0.7
        )
        sid_b = self.graph.find_similar_node(
            strategy_b_content, NodeType.STRATEGY, threshold=0.7
        )

        if not sid_a or not sid_b or sid_a == sid_b:
            return []

        ops = [
            GraphOperation(
                op_type="ADD_EDGE",
                source_id=sid_a,
                target_id=sid_b,
                edge_type=EdgeType.CONFLICTS_WITH.value,
            ),
            GraphOperation(
                op_type="ADD_EDGE",
                source_id=sid_b,
                target_id=sid_a,
                edge_type=EdgeType.CONFLICTS_WITH.value,
            ),
        ]
        return ops
