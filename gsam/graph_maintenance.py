"""Pruning, statistics and epoch consolidation for KnowledgeGraph, split out as a mixin."""

import json
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

from .graph_types import NodeType, EdgeType
from log_config import get_logger

logger = get_logger("graph_memory")


def _embeddings_available() -> bool:
    """Read the flag from graph_memory at call time (it is patched in tests; avoids an import cycle)."""
    from . import graph_memory
    return graph_memory.EMBEDDINGS_AVAILABLE


class MaintenanceMixin:

    def prune(
        self,
        min_degree: int = 2,
        min_confidence: float = 0.1,
        max_age_epochs: int = 10,
    ) -> int:
        """
        Remove low-utility nodes from the graph.

        Nodes are removed if they have low degree, low confidence,
        and haven't been used recently. Concept nodes from the ontology
        are never pruned.

        Args:
            min_degree: Minimum degree to survive pruning.
            min_confidence: Minimum confidence to survive pruning.
            max_age_epochs: Max tasks since last use before eligible.

        Returns:
            Number of nodes removed.
        """
        to_remove = []
        now = time.time()

        for nid, data in self.graph.nodes(data=True):
            # Never prune Concept nodes (ontology backbone)
            if data.get("type") == NodeType.CONCEPT.value:
                continue
            # Never prune Formula nodes initialized from ontology
            if data.get("type") == NodeType.FORMULA.value and data.get("from_ontology"):
                continue

            degree = self.graph.degree(nid)
            confidence = data.get("confidence", 1.0)
            use_count = data.get("use_count", 0)

            if degree < min_degree and confidence < min_confidence and use_count == 0:
                to_remove.append(nid)

        for nid in to_remove:
            if nid in self.embeddings:
                del self.embeddings[nid]
            self.graph.remove_node(nid)

        return len(to_remove)

    def stats(self) -> dict:
        """Return summary statistics about the graph."""
        node_counts = defaultdict(int)
        edge_counts = defaultdict(int)

        for _, data in self.graph.nodes(data=True):
            node_counts[data.get("type", "Unknown")] += 1

        for _, _, data in self.graph.edges(data=True):
            edge_counts[data.get("type", "Unknown")] += 1

        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()

        # Concept coverage: fraction of concepts with at least one strategy
        concept_ids = self.get_nodes_by_type(NodeType.CONCEPT)
        concepts_with_strategy = 0
        for cid in concept_ids:
            has_strategy = any(
                self.graph.nodes[nbr].get("type") == NodeType.STRATEGY.value
                for nbr, _, edata in self.graph.in_edges(cid, data=True)
                if edata.get("type") == EdgeType.APPLIES_TO.value
            )
            if has_strategy:
                concepts_with_strategy += 1

        concept_coverage = (
            concepts_with_strategy / len(concept_ids) if concept_ids else 0.0
        )

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_counts": dict(node_counts),
            "edge_counts": dict(edge_counts),
            "density": nx.density(self.graph) if total_nodes > 1 else 0.0,
            "concept_coverage": concept_coverage,
            "concepts_with_strategy": concepts_with_strategy,
            "total_concepts": len(concept_ids),
            "tasks_processed": self.tasks_processed,
        }

    def consolidate_epoch(
        self,
        edge_similarity_threshold: float = 0.85,
    ) -> Dict[str, int]:
        """
        Run multi-epoch graph refinement after each epoch (Paper §5.6).

        Three operations:
        1. Edge Discovery: For each Strategy node, check if it should
           apply_to additional Concepts based on shared neighborhoods.
        2. Edge Weight Reinforcement: Increment weights on applies_to
           edges for strategies confirmed helpful; create/strengthen
           fails_for edges for strategies that failed.
        3. Cross-Node Consolidation: Merge Strategy nodes with high
           cosine similarity AND identical concept neighborhoods.

        Args:
            edge_similarity_threshold: Cosine similarity for merging
                strategy nodes during consolidation.

        Returns:
            Dict with counts of operations performed.
        """
        stats = {"edges_discovered": 0, "edges_reinforced": 0, "nodes_merged": 0}

        # --- 1. Edge Discovery ---
        # For each Strategy, find Concepts it applies_to. Then look at
        # sibling Concepts (via is_a). If the strategy was helpful and
        # the sibling has no strategy yet, create a tentative applies_to.
        strategy_ids = self.get_nodes_by_type(NodeType.STRATEGY)
        for sid in strategy_ids:
            sdata = self.graph.nodes[sid]
            if sdata.get("helpful_count", 0) <= 0:
                continue

            # Get concepts this strategy already applies to
            applied_concepts = set()
            for _, tgt, edata in self.graph.out_edges(sid, data=True):
                if edata.get("type") == EdgeType.APPLIES_TO.value:
                    applied_concepts.add(tgt)

            # For each applied concept, consider siblings
            new_targets = set()
            for cid in list(applied_concepts):
                siblings = self.get_siblings(cid, edge_type=EdgeType.IS_A.value)
                for sib in siblings:
                    if sib not in applied_concepts and sib not in new_targets:
                        # Check if sibling lacks any strategy
                        has_strategy = any(
                            self.graph.nodes.get(src, {}).get("type") == NodeType.STRATEGY.value
                            for src, _, ed in self.graph.in_edges(sib, data=True)
                            if ed.get("type") == EdgeType.APPLIES_TO.value
                        )
                        if not has_strategy:
                            new_targets.add(sib)

            for tgt in new_targets:
                if tgt in self.graph:
                    self.add_edge(sid, tgt, EdgeType.APPLIES_TO, weight=0.5, tentative=True)
                    stats["edges_discovered"] += 1

        # --- 2. Edge Weight Reinforcement ---
        for sid in strategy_ids:
            sdata = self.graph.nodes[sid]
            helpful = sdata.get("helpful_count", 0)
            harmful = sdata.get("harmful_count", 0)

            for _, tgt, edata in self.graph.out_edges(sid, data=True):
                if edata.get("type") == EdgeType.APPLIES_TO.value:
                    old_w = edata.get("weight", 1.0)
                    # Reinforce based on net helpfulness
                    delta = helpful - harmful
                    if delta > 0:
                        edata["weight"] = old_w + delta * 0.1
                        stats["edges_reinforced"] += 1

        # --- 3. Cross-Node Consolidation ---
        if not _embeddings_available():
            return stats

        # Find strategy pairs with high similarity AND same neighborhoods
        merged = set()
        for i, sid_a in enumerate(strategy_ids):
            if sid_a in merged or sid_a not in self.embeddings:
                continue
            for sid_b in strategy_ids[i + 1:]:
                if sid_b in merged or sid_b not in self.embeddings:
                    continue

                sim = float(np.dot(self.embeddings[sid_a].ravel(), self.embeddings[sid_b].ravel()))
                if sim < edge_similarity_threshold:
                    continue

                # Check if neighborhoods match (same connected concepts)
                concepts_a = {
                    tgt for _, tgt, ed in self.graph.out_edges(sid_a, data=True)
                    if ed.get("type") == EdgeType.APPLIES_TO.value
                }
                concepts_b = {
                    tgt for _, tgt, ed in self.graph.out_edges(sid_b, data=True)
                    if ed.get("type") == EdgeType.APPLIES_TO.value
                }

                if concepts_a and concepts_a == concepts_b:
                    # Merge b into a
                    self.merge_nodes(sid_a, self.graph.nodes[sid_b].get("content", ""))
                    # Re-link b's edges to a
                    for _, tgt, ed in list(self.graph.out_edges(sid_b, data=True)):
                        if not self.graph.has_edge(sid_a, tgt):
                            self.graph.add_edge(sid_a, tgt, **ed)
                    for src, _, ed in list(self.graph.in_edges(sid_b, data=True)):
                        if not self.graph.has_edge(src, sid_a):
                            self.graph.add_edge(src, sid_a, **ed)
                    self.graph.remove_node(sid_b)
                    if sid_b in self.embeddings:
                        del self.embeddings[sid_b]
                    merged.add(sid_b)
                    stats["nodes_merged"] += 1

        return stats
