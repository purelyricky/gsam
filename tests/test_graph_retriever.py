"""Unit tests for GraphRetriever (BFS depth and node capping)."""

import unittest

from tests.helpers import make_graph, NodeType, EdgeType
from gsam.core.graph_retriever import GraphRetriever


class TestGraphRetriever(unittest.TestCase):
    def setUp(self):
        self.kg = make_graph()
        self.revenue = self.kg.add_node(NodeType.CONCEPT, "Revenue", name="Revenue")
        self.strategy = self.kg.add_node(NodeType.STRATEGY, "Use reported revenue")
        self.anti = self.kg.add_node(NodeType.ANTI_PATTERN, "Do not use net income")
        self.far_concept = self.kg.add_node(NodeType.CONCEPT, "Goodwill", name="Goodwill")
        # revenue <-applies_to- strategy -fixes-> anti -fails_for-> far_concept
        self.kg.add_edge(self.strategy, self.revenue, EdgeType.APPLIES_TO)
        self.kg.add_edge(self.strategy, self.anti, EdgeType.FIXES)
        self.kg.add_edge(self.anti, self.far_concept, EdgeType.FAILS_FOR)

    def test_no_matching_concept_returns_empty(self):
        text, ids = GraphRetriever(self.kg).retrieve("unrelated question")
        self.assertEqual(ids, [])
        self.assertIn("No relevant graph context", text)

    def test_keyword_match_retrieves_neighbours(self):
        _, ids = GraphRetriever(self.kg, retrieval_depth=2).retrieve("What is Revenue?")
        self.assertIn(self.revenue, ids)
        self.assertIn(self.strategy, ids)
        self.assertIn(self.anti, ids)

    def test_depth_limits_traversal(self):
        _, shallow = GraphRetriever(self.kg, retrieval_depth=1).retrieve("What is Revenue?")
        self.assertIn(self.strategy, shallow)
        self.assertNotIn(self.anti, shallow)

    def test_concept_nodes_are_capped(self):
        for i in range(15):
            cid = self.kg.add_node(NodeType.CONCEPT, f"Extra{i}", name=f"Extra{i}")
            self.kg.add_edge(self.strategy, cid, EdgeType.APPLIES_TO)
        _, ids = GraphRetriever(self.kg, retrieval_depth=2).retrieve("What is Revenue?")
        concept_ids = [n for n in ids if self.kg.graph.nodes[n]["type"] == NodeType.CONCEPT.value]
        self.assertLessEqual(len(concept_ids), 10)
        self.assertIn(self.strategy, ids)

    def test_knowledge_nodes_are_capped(self):
        for i in range(30):
            sid = self.kg.add_node(NodeType.STRATEGY, f"Strategy {i}", helpful_count=i)
            self.kg.add_edge(sid, self.revenue, EdgeType.APPLIES_TO)
        _, ids = GraphRetriever(self.kg, retrieval_depth=1).retrieve("Revenue")
        knowledge = [n for n in ids if self.kg.graph.nodes[n]["type"] != NodeType.CONCEPT.value]
        self.assertLessEqual(len(knowledge), 20)

    def test_serialized_output_mentions_strategy(self):
        text, _ = GraphRetriever(self.kg).retrieve("Revenue")
        self.assertIn("Use reported revenue", text)

    def test_identify_concepts_respects_max_concepts(self):
        for i in range(5):
            self.kg.add_node(NodeType.CONCEPT, f"Item{i}", name=f"Item{i}")
        retriever = GraphRetriever(self.kg, max_concepts=2)
        matches = retriever._identify_concepts("Revenue Goodwill Item0 Item1 Item2")
        self.assertEqual(len(matches), 2)


if __name__ == "__main__":
    unittest.main()
