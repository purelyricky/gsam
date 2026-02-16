"""
GSAM (Graph-Structured Adaptive Memory)

Extends ACE with ontology-grounded knowledge graph storage,
replacing flat bullet-point storage with typed nodes and edges.

Key components:
- KnowledgeGraph: Graph-structured memory (replaces playbook_utils.py)
- GraphConstructor: Converts curator deltas into graph operations
- GraphRetriever: Ontology-aware subgraph retrieval
- OntologyLoader: XBRL taxonomy initialization
- GSAM: Main orchestrator (extends ACE)
"""

from .gsam import GSAM
from .graph_memory import KnowledgeGraph, NodeType, EdgeType

__all__ = ['GSAM', 'KnowledgeGraph', 'NodeType', 'EdgeType']

__version__ = "0.1.0"
