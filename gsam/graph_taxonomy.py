"""Taxonomy (is_a) navigation helpers for KnowledgeGraph, split out as a mixin."""

import json
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

from .graph_types import NodeType, EdgeType
from log_config import get_logger

logger = get_logger("graph_memory")


class TaxonomyMixin:

    def get_ancestors(self, node_id: str, edge_type: str = "is_a") -> List[str]:
        """Walk is_a edges upward to get all ancestors."""
        ancestors = []
        current = node_id
        visited = set()
        while current and current not in visited:
            visited.add(current)
            parents = [
                tgt for _, tgt, edata in self.graph.out_edges(current, data=True)
                if edata.get("type") == edge_type
            ]
            if parents:
                ancestors.append(parents[0])
                current = parents[0]
            else:
                break
        return ancestors

    def get_children(self, node_id: str, edge_type: str = "is_a") -> List[str]:
        """Get direct children via is_a edges."""
        return [
            src for src, _, edata in self.graph.in_edges(node_id, data=True)
            if edata.get("type") == edge_type
        ]

    def get_siblings(self, node_id: str, edge_type: str = "is_a") -> List[str]:
        """Get sibling nodes (share a direct parent via is_a)."""
        parents = [
            tgt for _, tgt, edata in self.graph.out_edges(node_id, data=True)
            if edata.get("type") == edge_type
        ]
        siblings = []
        for parent in parents:
            for child in self.get_children(parent, edge_type):
                if child != node_id:
                    siblings.append(child)
        return siblings

    def lowest_common_ancestor_depth(
        self, node_a: str, node_b: str, edge_type: str = "is_a"
    ) -> int:
        """
        Compute depth of lowest common ancestor in taxonomy.

        Returns -1 if no common ancestor exists.
        """
        ancestors_a = [node_a] + self.get_ancestors(node_a, edge_type)
        ancestors_b_set = set([node_b] + self.get_ancestors(node_b, edge_type))

        for depth, anc in enumerate(ancestors_a):
            if anc in ancestors_b_set:
                return depth
        return -1
