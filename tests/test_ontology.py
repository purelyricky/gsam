"""
Unit tests for GSAM ontology loader (ontology.py).

Tests taxonomy parsing, concept node creation, and entity mapping.
"""

import json
import os
import sys
import tempfile
import unittest
import importlib

# Import modules directly to avoid gsam/__init__.py which triggers
# heavy imports (tiktoken, openai, etc.)
_gm_spec = importlib.util.spec_from_file_location(
    "gsam.graph_memory",
    os.path.join(os.path.dirname(__file__), "..", "gsam", "graph_memory.py"),
)
_gm_mod = importlib.util.module_from_spec(_gm_spec)
sys.modules["gsam.graph_memory"] = _gm_mod
_gm_spec.loader.exec_module(_gm_mod)
KnowledgeGraph = _gm_mod.KnowledgeGraph
NodeType = _gm_mod.NodeType
EdgeType = _gm_mod.EdgeType

# Now import ontology (which imports from gsam.graph_memory)
_ont_spec = importlib.util.spec_from_file_location(
    "gsam.ontology",
    os.path.join(os.path.dirname(__file__), "..", "gsam", "ontology.py"),
)
_ont_mod = importlib.util.module_from_spec(_ont_spec)
_ont_spec.loader.exec_module(_ont_mod)
initialize_ontology = _ont_mod.initialize_ontology
get_entity_name_to_node_map = _ont_mod.get_entity_name_to_node_map


SAMPLE_TAXONOMY = {
    "taxonomy_name": "Test-Taxonomy",
    "version": "2024",
    "categories": {
        "RevenueAndIncome": {
            "description": "Revenue-related elements",
            "children": {
                "RevenueRecognition": {
                    "description": "Revenue recognition entities",
                    "entities": [
                        "Revenues",
                        "RevenueFromContractWithCustomerExcludingAssessedTax"
                    ]
                },
                "GainsAndLosses": {
                    "description": "Gains and losses",
                    "entities": [
                        "GainsLossesOnExtinguishmentOfDebt"
                    ]
                }
            }
        },
        "ExpensesAndCosts": {
            "description": "Expense-related elements",
            "children": {
                "DepreciationAndAmortization": {
                    "description": "Depreciation and amortization",
                    "entities": [
                        "Depreciation",
                        "AmortizationOfIntangibleAssets"
                    ]
                }
            }
        }
    },
    "entity_definitions": {
        "Revenues": {
            "description": "Total revenues from all sources",
            "category_path": "RevenueAndIncome > RevenueRecognition"
        },
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "description": "Revenue from contracts excluding taxes",
            "category_path": "RevenueAndIncome > RevenueRecognition"
        },
        "GainsLossesOnExtinguishmentOfDebt": {
            "description": "Gains or losses from debt extinguishment",
            "category_path": "RevenueAndIncome > GainsAndLosses"
        },
        "Depreciation": {
            "description": "Depreciation of tangible assets",
            "category_path": "ExpensesAndCosts > DepreciationAndAmortization"
        },
        "AmortizationOfIntangibleAssets": {
            "description": "Amortization of intangible assets",
            "category_path": "ExpensesAndCosts > DepreciationAndAmortization"
        }
    }
}


class TestOntologyInitialization(unittest.TestCase):
    """Test ontology loading and graph initialization."""

    def setUp(self):
        self.kg = KnowledgeGraph.__new__(KnowledgeGraph)
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

        # Write sample taxonomy to temp file
        self.tax_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(SAMPLE_TAXONOMY, self.tax_file)
        self.tax_file.close()

    def tearDown(self):
        os.unlink(self.tax_file.name)

    def test_loads_entities(self):
        entity_map = initialize_ontology(self.kg, self.tax_file.name)
        self.assertEqual(len(entity_map), 5)
        self.assertIn("Revenues", entity_map)
        self.assertIn("Depreciation", entity_map)

    def test_creates_category_nodes(self):
        initialize_ontology(self.kg, self.tax_file.name)
        concepts = self.kg.get_nodes_by_type(NodeType.CONCEPT)
        # 2 categories + 3 subcategories + 5 entities = 10
        self.assertEqual(len(concepts), 10)

    def test_creates_is_a_edges(self):
        entity_map = initialize_ontology(self.kg, self.tax_file.name)
        revenues_id = entity_map["Revenues"]
        # Revenues should have is_a edge to RevenueRecognition subcategory
        parents = [
            tgt for _, tgt, edata in self.kg.graph.out_edges(revenues_id, data=True)
            if edata.get("type") == EdgeType.IS_A.value
        ]
        self.assertEqual(len(parents), 1)

    def test_taxonomy_path(self):
        entity_map = initialize_ontology(self.kg, self.tax_file.name)
        revenues_id = entity_map["Revenues"]
        data = self.kg.graph.nodes[revenues_id]
        self.assertEqual(data["taxonomy_path"],
                         "RevenueAndIncome > RevenueRecognition")

    def test_siblings_within_subcategory(self):
        entity_map = initialize_ontology(self.kg, self.tax_file.name)
        revenues_id = entity_map["Revenues"]
        siblings = self.kg.get_siblings(revenues_id, EdgeType.IS_A.value)
        # Should find RevenueFromContractWithCustomerExcludingAssessedTax
        sibling_names = []
        for sid in siblings:
            sibling_names.append(self.kg.graph.nodes[sid].get("name", ""))
        self.assertIn(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            sibling_names
        )


class TestGetEntityNameMap(unittest.TestCase):
    """Test entity name to node ID mapping."""

    def setUp(self):
        self.kg = KnowledgeGraph.__new__(KnowledgeGraph)
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

        self.tax_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(SAMPLE_TAXONOMY, self.tax_file)
        self.tax_file.close()

    def tearDown(self):
        os.unlink(self.tax_file.name)

    def test_get_entity_map(self):
        initialize_ontology(self.kg, self.tax_file.name)
        entity_map = get_entity_name_to_node_map(self.kg)
        self.assertEqual(len(entity_map), 5)
        for name, nid in entity_map.items():
            self.assertIn(nid, self.kg.graph)
            data = self.kg.graph.nodes[nid]
            self.assertTrue(data.get("is_entity"))


if __name__ == "__main__":
    unittest.main()
