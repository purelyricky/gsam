"""Shared offline fixtures: a KnowledgeGraph that never loads an embedding model."""

import os
import sys
import time
from collections import defaultdict

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gsam import graph_memory  # noqa: E402
from gsam.graph_memory import KnowledgeGraph, NodeType, EdgeType  # noqa: E402,F401

# Force the keyword-only code paths so tests need no network / model download.
graph_memory.EMBEDDINGS_AVAILABLE = False


def make_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph.__new__(KnowledgeGraph)
    kg.graph = nx.DiGraph()
    kg.embeddings = {}
    kg.embedding_model_name = "all-MiniLM-L6-v2"
    kg._embedding_model = None
    kg._next_ids = defaultdict(int)
    kg.created_at = time.time()
    kg.tasks_processed = 0
    kg.use_typed_edges = True
    return kg
