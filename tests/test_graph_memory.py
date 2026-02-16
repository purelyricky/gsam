"""
Unit tests for GSAM KnowledgeGraph (graph_memory.py).

Tests node/edge CRUD, similarity-based dedup, subgraph extraction,
serialization/deserialization, taxonomy helpers, and pruning.
"""

import json
import os
import sys
import tempfile
import unittest
import importlib

# Import graph_memory directly to avoid gsam/__init__.py which triggers
# heavy imports (tiktoken, openai, etc.)
_spec = importlib.util.spec_from_file_location(
    "gsam.graph_memory",
    os.path.join(os.path.dirname(__file__), "..", "gsam", "graph_memory.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["gsam.graph_memory"] = _mod
_spec.loader.exec_module(_mod)
KnowledgeGraph = _mod.KnowledgeGraph
NodeType = _mod.NodeType
EdgeType = _mod.EdgeType


class TestKnowledgeGraphBasic(unittest.TestCase):
    """Test basic node and edge operations."""

    def setUp(self):
        self.kg = KnowledgeGraph.__new__(KnowledgeGraph)
        # Manually init without loading embedding model
        import networkx as nx
        from collections import defaultdict
        import time
        self.kg.graph = nx.DiGraph()
        self.kg.embeddings = {}
        self.kg.embedding_model_name = "all-MiniLM-L6-v2"
        self.kg._embedding_model = None
        self.kg._next_ids = defaultdict(int)
        self.kg.created_at = time.time()
        self.kg.tasks_processed = 0
        self.kg.use_typed_edges = True

    def test_add_node_auto_id(self):
        nid = self.kg.add_node(NodeType.STRATEGY, "Test strategy content")
        self.assertEqual(nid, "S:0001")
        self.assertIn(nid, self.kg.graph)
        self.assertEqual(self.kg.graph.nodes[nid]["type"], "Strategy")
        self.assertEqual(self.kg.graph.nodes[nid]["content"], "Test strategy content")

    def test_add_node_explicit_id(self):
        nid = self.kg.add_node(NodeType.CONCEPT, "Revenue", node_id="C:0042")
        self.assertEqual(nid, "C:0042")
        self.assertIn("C:0042", self.kg.graph)

    def test_add_node_increments_id(self):
        n1 = self.kg.add_node(NodeType.STRATEGY, "First")
        n2 = self.kg.add_node(NodeType.STRATEGY, "Second")
        self.assertEqual(n1, "S:0001")
        self.assertEqual(n2, "S:0002")

    def test_add_node_different_types(self):
        s = self.kg.add_node(NodeType.STRATEGY, "strat")
        a = self.kg.add_node(NodeType.ANTI_PATTERN, "anti")
        c = self.kg.add_node(NodeType.CONCEPT, "concept")
        f = self.kg.add_node(NodeType.FORMULA, "formula")
        x = self.kg.add_node(NodeType.CONFUSION, "confusion")
        self.assertTrue(s.startswith("S:"))
        self.assertTrue(a.startswith("A:"))
        self.assertTrue(c.startswith("C:"))
        self.assertTrue(f.startswith("F:"))
        self.assertTrue(x.startswith("X:"))

    def test_add_node_default_attrs(self):
        nid = self.kg.add_node(NodeType.STRATEGY, "test")
        data = self.kg.graph.nodes[nid]
        self.assertEqual(data["helpful_count"], 0)
        self.assertEqual(data["harmful_count"], 0)
        self.assertEqual(data["confidence"], 1.0)
        self.assertEqual(data["use_count"], 0)

    def test_add_node_custom_attrs(self):
        nid = self.kg.add_node(NodeType.STRATEGY, "test", helpful_count=5)
        self.assertEqual(self.kg.graph.nodes[nid]["helpful_count"], 5)

    def test_add_edge(self):
        s = self.kg.add_node(NodeType.STRATEGY, "strat")
        c = self.kg.add_node(NodeType.CONCEPT, "concept")
        self.kg.add_edge(s, c, EdgeType.APPLIES_TO)
        self.assertTrue(self.kg.graph.has_edge(s, c))
        edata = self.kg.graph.edges[s, c]
        self.assertEqual(edata["type"], "applies_to")

    def test_add_edge_invalid_nodes(self):
        s = self.kg.add_node(NodeType.STRATEGY, "strat")
        with self.assertRaises(ValueError):
            self.kg.add_edge(s, "nonexistent", EdgeType.APPLIES_TO)

    def test_add_edge_untyped(self):
        self.kg.use_typed_edges = False
        s = self.kg.add_node(NodeType.STRATEGY, "strat")
        c = self.kg.add_node(NodeType.CONCEPT, "concept")
        self.kg.add_edge(s, c, EdgeType.APPLIES_TO)
        edata = self.kg.graph.edges[s, c]
        self.assertEqual(edata["type"], "related_to")

    def test_merge_nodes(self):
        nid = self.kg.add_node(NodeType.STRATEGY, "original content", helpful_count=2)
        self.kg.merge_nodes(nid, "extra detail", helpful_count=3)
        data = self.kg.graph.nodes[nid]
        self.assertIn("extra detail", data["content"])
        self.assertEqual(data["helpful_count"], 5)

    def test_merge_nodes_same_content(self):
        nid = self.kg.add_node(NodeType.STRATEGY, "same content")
        self.kg.merge_nodes(nid, "same content")
        # Content should not duplicate
        self.assertEqual(self.kg.graph.nodes[nid]["content"], "same content")

    def test_update_node_attr(self):
        nid = self.kg.add_node(NodeType.STRATEGY, "test")
        self.kg.update_node_attr(nid, helpful_count=10)
        self.assertEqual(self.kg.graph.nodes[nid]["helpful_count"], 10)

    def test_update_node_attr_nonexistent(self):
        # Should not raise
        self.kg.update_node_attr("nonexistent", helpful_count=10)


class TestKnowledgeGraphRetrieval(unittest.TestCase):
    """Test subgraph and neighbor retrieval."""

    def setUp(self):
        import networkx as nx
        from collections import defaultdict
        import time
        self.kg = KnowledgeGraph.__new__(KnowledgeGraph)
        self.kg.graph = nx.DiGraph()
        self.kg.embeddings = {}
        self.kg.embedding_model_name = "all-MiniLM-L6-v2"
        self.kg._embedding_model = None
        self.kg._next_ids = defaultdict(int)
        self.kg.created_at = time.time()
        self.kg.tasks_processed = 0
        self.kg.use_typed_edges = True

        # Build a small graph
        self.c1 = self.kg.add_node(NodeType.CONCEPT, "Revenue", name="Revenue")
        self.c2 = self.kg.add_node(NodeType.CONCEPT, "Expenses", name="Expenses")
        self.s1 = self.kg.add_node(NodeType.STRATEGY, "Check revenue recognition")
        self.a1 = self.kg.add_node(NodeType.ANTI_PATTERN, "Confusing gross and net")
        self.kg.add_edge(self.s1, self.c1, EdgeType.APPLIES_TO)
        self.kg.add_edge(self.a1, self.c1, EdgeType.FAILS_FOR)

    def test_get_nodes_by_type(self):
        concepts = self.kg.get_nodes_by_type(NodeType.CONCEPT)
        self.assertEqual(len(concepts), 2)
        strategies = self.kg.get_nodes_by_type(NodeType.STRATEGY)
        self.assertEqual(len(strategies), 1)

    def test_get_neighbors_out(self):
        neighbors = self.kg.get_neighbors(self.s1, direction="out")
        neighbor_ids = [n[0] for n in neighbors]
        self.assertIn(self.c1, neighbor_ids)

    def test_get_neighbors_in(self):
        neighbors = self.kg.get_neighbors(self.c1, direction="in")
        neighbor_ids = [n[0] for n in neighbors]
        self.assertIn(self.s1, neighbor_ids)
        self.assertIn(self.a1, neighbor_ids)

    def test_get_neighbors_filtered(self):
        neighbors = self.kg.get_neighbors(
            self.c1, edge_type=EdgeType.APPLIES_TO, direction="in"
        )
        neighbor_ids = [n[0] for n in neighbors]
        self.assertIn(self.s1, neighbor_ids)
        self.assertNotIn(self.a1, neighbor_ids)

    def test_get_subgraph(self):
        sub = self.kg.get_subgraph([self.c1], depth=1)
        self.assertIn(self.c1, sub.nodes)
        self.assertIn(self.s1, sub.nodes)
        self.assertIn(self.a1, sub.nodes)
        # c2 should not be reached from c1
        self.assertNotIn(self.c2, sub.nodes)

    def test_get_subgraph_depth_0(self):
        sub = self.kg.get_subgraph([self.c1], depth=0)
        self.assertIn(self.c1, sub.nodes)
        self.assertEqual(len(sub.nodes), 1)

    def test_get_subgraph_edge_type_filter(self):
        sub = self.kg.get_subgraph(
            [self.c1], depth=1,
            edge_types={EdgeType.APPLIES_TO.value}
        )
        self.assertIn(self.s1, sub.nodes)
        self.assertNotIn(self.a1, sub.nodes)

    def test_find_concepts_by_name(self):
        matches = self.kg.find_concepts_by_name("What is Revenue?")
        self.assertIn(self.c1, matches)


class TestKnowledgeGraphTaxonomy(unittest.TestCase):
    """Test taxonomy helper methods."""

    def setUp(self):
        import networkx as nx
        from collections import defaultdict
        import time
        self.kg = KnowledgeGraph.__new__(KnowledgeGraph)
        self.kg.graph = nx.DiGraph()
        self.kg.embeddings = {}
        self.kg.embedding_model_name = "all-MiniLM-L6-v2"
        self.kg._embedding_model = None
        self.kg._next_ids = defaultdict(int)
        self.kg.created_at = time.time()
        self.kg.tasks_processed = 0
        self.kg.use_typed_edges = True

        # Build taxonomy: Root -> Cat -> SubA, SubB
        self.root = self.kg.add_node(NodeType.CONCEPT, "FinancialStatements", name="Root")
        self.cat = self.kg.add_node(NodeType.CONCEPT, "IncomeStatement", name="IncomeStatement")
        self.sub_a = self.kg.add_node(NodeType.CONCEPT, "Revenue", name="Revenue")
        self.sub_b = self.kg.add_node(NodeType.CONCEPT, "Expenses", name="Expenses")

        self.kg.add_edge(self.cat, self.root, EdgeType.IS_A)
        self.kg.add_edge(self.sub_a, self.cat, EdgeType.IS_A)
        self.kg.add_edge(self.sub_b, self.cat, EdgeType.IS_A)

    def test_get_ancestors(self):
        ancestors = self.kg.get_ancestors(self.sub_a)
        self.assertEqual(ancestors, [self.cat, self.root])

    def test_get_children(self):
        children = self.kg.get_children(self.cat)
        self.assertIn(self.sub_a, children)
        self.assertIn(self.sub_b, children)

    def test_get_siblings(self):
        siblings = self.kg.get_siblings(self.sub_a)
        self.assertIn(self.sub_b, siblings)
        self.assertNotIn(self.sub_a, siblings)

    def test_lca_depth(self):
        depth = self.kg.lowest_common_ancestor_depth(self.sub_a, self.sub_b)
        # sub_a -> cat (depth 1), sub_b is under cat too
        self.assertEqual(depth, 1)

    def test_lca_depth_no_common(self):
        orphan = self.kg.add_node(NodeType.CONCEPT, "Orphan")
        depth = self.kg.lowest_common_ancestor_depth(self.sub_a, orphan)
        self.assertEqual(depth, -1)


class TestKnowledgeGraphSerialization(unittest.TestCase):
    """Test serialization and deserialization."""

    def setUp(self):
        import networkx as nx
        from collections import defaultdict
        import time
        self.kg = KnowledgeGraph.__new__(KnowledgeGraph)
        self.kg.graph = nx.DiGraph()
        self.kg.embeddings = {}
        self.kg.embedding_model_name = "all-MiniLM-L6-v2"
        self.kg._embedding_model = None
        self.kg._next_ids = defaultdict(int)
        self.kg.created_at = time.time()
        self.kg.tasks_processed = 5
        self.kg.use_typed_edges = True

    def test_to_json_and_back(self):
        c = self.kg.add_node(NodeType.CONCEPT, "Revenue", name="Revenue")
        s = self.kg.add_node(NodeType.STRATEGY, "Check recognition")
        self.kg.add_edge(s, c, EdgeType.APPLIES_TO)

        data = self.kg.to_json()
        self.assertEqual(len(data["nodes"]), 2)
        self.assertEqual(len(data["edges"]), 1)

        # Reconstruct
        kg2 = KnowledgeGraph.from_json(data)
        self.assertEqual(kg2.graph.number_of_nodes(), 2)
        self.assertEqual(kg2.graph.number_of_edges(), 1)
        self.assertEqual(kg2.tasks_processed, 5)

    def test_save_and_load(self):
        c = self.kg.add_node(NodeType.CONCEPT, "Test", name="Test")
        s = self.kg.add_node(NodeType.STRATEGY, "Test strategy")
        self.kg.add_edge(s, c, EdgeType.APPLIES_TO)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            self.kg.save(path)
            loaded = KnowledgeGraph.load(path)
            self.assertEqual(loaded.graph.number_of_nodes(), 2)
            self.assertEqual(loaded.graph.number_of_edges(), 1)
        finally:
            os.unlink(path)

    def test_serialize_subgraph(self):
        c = self.kg.add_node(NodeType.CONCEPT, "Revenue", name="Revenue",
                             taxonomy_path="Income > Revenue")
        s = self.kg.add_node(NodeType.STRATEGY, "Check timing",
                             helpful_count=3, harmful_count=0, confidence=0.9)
        self.kg.add_edge(s, c, EdgeType.APPLIES_TO)
        sub = self.kg.get_subgraph([c], depth=1)
        text = self.kg.serialize_subgraph(sub)
        self.assertIn("RELEVANT CONCEPTS", text)
        self.assertIn("Revenue", text)
        self.assertIn("Check timing", text)

    def test_serialize_empty_subgraph(self):
        import networkx as nx
        empty = nx.DiGraph()
        text = self.kg.serialize_subgraph(empty)
        self.assertIn("No relevant graph context", text)


class TestKnowledgeGraphPruning(unittest.TestCase):
    """Test node pruning."""

    def setUp(self):
        import networkx as nx
        from collections import defaultdict
        import time
        self.kg = KnowledgeGraph.__new__(KnowledgeGraph)
        self.kg.graph = nx.DiGraph()
        self.kg.embeddings = {}
        self.kg.embedding_model_name = "all-MiniLM-L6-v2"
        self.kg._embedding_model = None
        self.kg._next_ids = defaultdict(int)
        self.kg.created_at = time.time()
        self.kg.tasks_processed = 0
        self.kg.use_typed_edges = True

    def test_prune_removes_low_utility(self):
        # Concept nodes should never be pruned
        c = self.kg.add_node(NodeType.CONCEPT, "Revenue", confidence=0.01, use_count=0)

        # Low-utility strategy: no edges, low confidence, unused
        s = self.kg.add_node(NodeType.STRATEGY, "bad strategy", confidence=0.01, use_count=0)

        pruned = self.kg.prune(min_degree=1, min_confidence=0.1)
        self.assertEqual(pruned, 1)  # Only strategy pruned
        self.assertIn(c, self.kg.graph)
        self.assertNotIn(s, self.kg.graph)

    def test_prune_keeps_connected(self):
        c = self.kg.add_node(NodeType.CONCEPT, "Revenue")
        s = self.kg.add_node(NodeType.STRATEGY, "good strategy", confidence=0.01)
        self.kg.add_edge(s, c, EdgeType.APPLIES_TO)

        pruned = self.kg.prune(min_degree=1, min_confidence=0.1)
        self.assertEqual(pruned, 0)  # Has an edge, degree >= 1
        self.assertIn(s, self.kg.graph)

    def test_prune_keeps_ontology_formulas(self):
        f = self.kg.add_node(NodeType.FORMULA, "EPS formula",
                             from_ontology=True, confidence=0.01, use_count=0)
        pruned = self.kg.prune(min_degree=1, min_confidence=0.1)
        self.assertEqual(pruned, 0)
        self.assertIn(f, self.kg.graph)


class TestKnowledgeGraphStats(unittest.TestCase):
    """Test graph statistics."""

    def setUp(self):
        import networkx as nx
        from collections import defaultdict
        import time
        self.kg = KnowledgeGraph.__new__(KnowledgeGraph)
        self.kg.graph = nx.DiGraph()
        self.kg.embeddings = {}
        self.kg.embedding_model_name = "all-MiniLM-L6-v2"
        self.kg._embedding_model = None
        self.kg._next_ids = defaultdict(int)
        self.kg.created_at = time.time()
        self.kg.tasks_processed = 0
        self.kg.use_typed_edges = True

    def test_stats_empty(self):
        stats = self.kg.stats()
        self.assertEqual(stats["total_nodes"], 0)
        self.assertEqual(stats["total_edges"], 0)
        self.assertEqual(stats["concept_coverage"], 0.0)

    def test_stats_with_nodes(self):
        c = self.kg.add_node(NodeType.CONCEPT, "Revenue")
        s = self.kg.add_node(NodeType.STRATEGY, "Check revenue")
        self.kg.add_edge(s, c, EdgeType.APPLIES_TO)

        stats = self.kg.stats()
        self.assertEqual(stats["total_nodes"], 2)
        self.assertEqual(stats["total_edges"], 1)
        self.assertEqual(stats["total_concepts"], 1)
        self.assertEqual(stats["concepts_with_strategy"], 1)
        self.assertEqual(stats["concept_coverage"], 1.0)

    def test_repr(self):
        self.kg.add_node(NodeType.CONCEPT, "Test")
        r = repr(self.kg)
        self.assertIn("KnowledgeGraph", r)
        self.assertIn("nodes=1", r)


if __name__ == "__main__":
    unittest.main()
