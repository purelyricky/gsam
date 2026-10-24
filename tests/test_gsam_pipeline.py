"""Lightweight pipeline test: retriever -> generator with the LLM client stubbed out."""

import json
import unittest
from unittest import mock

from tests.helpers import make_graph, NodeType, EdgeType
from gsam import gsam as gsam_module
from gsam.core.graph_retriever import GraphRetriever


class TestGSAMPipeline(unittest.TestCase):
    def setUp(self):
        self.kg = make_graph()
        self.concept = self.kg.add_node(NodeType.CONCEPT, "Revenue", name="Revenue")
        self.strategy = self.kg.add_node(NodeType.STRATEGY, "Prefer reported revenue")
        self.kg.add_edge(self.strategy, self.concept, EdgeType.APPLIES_TO)

    def test_generate_end_to_end_with_stubbed_llm(self):
        graph_context, retrieved = GraphRetriever(self.kg).retrieve("What is Revenue?")
        self.assertIn(self.strategy, retrieved)

        reply = json.dumps({"final_answer": "42", "node_ids": [self.strategy]})
        with mock.patch.object(gsam_module, "timed_llm_call", return_value=(reply, {"role": "generator"})) as llm:
            gen = gsam_module.GSAMGenerator(api_client=None, api_provider="test", model="stub")
            response, node_ids, call_info = gen.generate(
                question="What is Revenue?", graph_context=graph_context, context="10-K excerpt",
            )

        self.assertEqual(response, reply)
        self.assertEqual(node_ids, [self.strategy])
        self.assertEqual(call_info["role"], "generator")
        prompt = llm.call_args[0][3]
        self.assertIn("Prefer reported revenue", prompt)
        self.assertIn("What is Revenue?", prompt)

    def test_node_ids_fall_back_to_regex(self):
        gen = gsam_module.GSAMGenerator(None, "test", "stub")
        self.assertEqual(gen._extract_node_ids("used [S:0001] and [A:0002]", False), ["S:0001", "A:0002"])

    def test_playbook_used_when_no_graph_context(self):
        with mock.patch.object(gsam_module, "timed_llm_call", return_value=("{}", {})) as llm:
            gsam_module.GSAMGenerator(None, "test", "stub").generate(question="q", playbook="PLAYBOOK-TEXT")
        self.assertIn("PLAYBOOK-TEXT", llm.call_args[0][3])


if __name__ == "__main__":
    unittest.main()
