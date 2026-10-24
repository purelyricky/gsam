"""Unit tests for GraphConstructor.process_curator_output with mocked curator JSON."""

import json
import unittest

from tests.helpers import make_graph, NodeType
from gsam.core.graph_constructor import GraphConstructor, GraphOperation


def curator(*ops):
    return json.dumps({"operations": list(ops)})


class TestGraphConstructor(unittest.TestCase):
    def setUp(self):
        self.kg = make_graph()
        self.rev = self.kg.add_node(NodeType.CONCEPT, "Revenue", name="Revenue")
        self.sales = self.kg.add_node(NodeType.CONCEPT, "Sales", name="Sales")
        self.gc = GraphConstructor(
            api_client=None, api_provider="test", model="stub", graph=self.kg,
            entity_name_map={"Revenue": self.rev, "Sales": self.sales},
        )

    def _process(self, output, reflection="(empty)"):
        return self.gc.process_curator_output(output, reflection, "task")

    def test_add_strategy_creates_node_and_edge(self):
        ops = self._process(curator({
            "type": "ADD_STRATEGY", "content": "Prefer XBRL tag", "concepts": ["Revenue"],
        }))
        self.assertTrue(any(o.op_type == "ADD_NODE" for o in ops))
        strategies = self.kg.get_nodes_by_type(NodeType.STRATEGY)
        self.assertEqual(len(strategies), 1)
        self.assertTrue(self.kg.graph.has_edge(strategies[0], self.rev))
        self.assertEqual(self.kg.graph.edges[strategies[0], self.rev]["type"], "applies_to")

    def test_add_strategy_without_content_is_ignored(self):
        ops = self._process(curator({"type": "ADD_STRATEGY", "content": ""}))
        self.assertEqual(ops, [])
        self.assertEqual(self.kg.get_nodes_by_type(NodeType.STRATEGY), [])

    def test_add_confusion_links_concept_pair_both_ways(self):
        self._process(curator({
            "type": "ADD_CONFUSION", "content": "Revenue vs Sales",
            "concept_pair": ["Revenue", "Sales"], "distinguishing_criteria": "gross vs net",
        }))
        self.assertEqual(len(self.kg.get_nodes_by_type(NodeType.CONFUSION)), 1)
        for a, b in ((self.rev, self.sales), (self.sales, self.rev)):
            self.assertEqual(self.kg.graph.edges[a, b]["type"], "confused_with")

    def test_add_confusion_requires_pair(self):
        ops = self._process(curator({"type": "ADD_CONFUSION", "content": "x", "concept_pair": ["Revenue"]}))
        self.assertEqual(ops, [])

    def test_add_antipattern_and_ablation_flag(self):
        op = {"type": "ADD_ANTIPATTERN", "content": "Used wrong tag", "concepts": ["Revenue"]}
        self._process(curator(op))
        self.assertEqual(len(self.kg.get_nodes_by_type(NodeType.ANTI_PATTERN)), 1)

        kg2 = make_graph()
        gc2 = GraphConstructor(None, "test", "stub", kg2, {}, no_failure_cascades=True)
        gc2.process_curator_output(curator(op), "(empty)", "task")
        self.assertEqual(kg2.get_nodes_by_type(NodeType.ANTI_PATTERN), [])

    def test_invalid_json_returns_no_operations(self):
        self.assertEqual(self._process("not json at all"), [])

    def test_resolve_concept_is_case_insensitive(self):
        self.assertEqual(self.gc._resolve_concept("revenue"), self.rev)
        self.assertIsNone(self.gc._resolve_concept("zzzz-unknown"))

    def test_update_attr_increments_counters(self):
        sid = self.kg.add_node(NodeType.STRATEGY, "s")
        self.gc._apply_operations([
            GraphOperation(op_type="UPDATE_ATTR", node_id=sid, attrs={"helpful_count": 2}),
            GraphOperation(op_type="UPDATE_ATTR", node_id=sid, attrs={"helpful_count": 1}),
        ])
        self.assertEqual(self.kg.graph.nodes[sid]["helpful_count"], 3)


if __name__ == "__main__":
    unittest.main()
