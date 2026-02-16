"""
GSAM Ontology Loader

Parses the XBRL taxonomy from a JSON file and initializes
Concept nodes and is_a edges in the knowledge graph.
Also extracts Formula nodes from the Formula training data.
"""

import json
import os
from typing import Dict, List, Optional, Any

from .graph_memory import KnowledgeGraph, NodeType, EdgeType


def initialize_ontology(
    graph: KnowledgeGraph,
    taxonomy_path: str,
    formula_data_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Initialize the knowledge graph with XBRL taxonomy structure.

    Creates Concept nodes for all entity types and is_a edges
    for parent-child relationships in the taxonomy.

    Args:
        graph: KnowledgeGraph instance to populate.
        taxonomy_path: Path to xbrl_taxonomy.json.
        formula_data_path: Optional path to formula training JSONL
            for extracting Formula nodes.

    Returns:
        Dict mapping entity names to node IDs.
    """
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    entity_to_node_id: Dict[str, str] = {}

    # Step 1: Create category and subcategory concept nodes
    categories = taxonomy.get("categories", {})
    entity_defs = taxonomy.get("entity_definitions", {})

    # Create top-level category nodes
    cat_node_ids: Dict[str, str] = {}
    for cat_name, cat_data in categories.items():
        cat_id = graph.add_node(
            node_type=NodeType.CONCEPT,
            content=cat_data.get("description", cat_name),
            name=cat_name,
            taxonomy_path=cat_name,
            is_category=True,
        )
        cat_node_ids[cat_name] = cat_id

        # Create subcategory nodes
        subcat_node_ids: Dict[str, str] = {}
        for subcat_name, subcat_data in cat_data.get("children", {}).items():
            subcat_id = graph.add_node(
                node_type=NodeType.CONCEPT,
                content=subcat_data.get("description", subcat_name),
                name=subcat_name,
                taxonomy_path=f"{cat_name} > {subcat_name}",
                is_subcategory=True,
            )
            subcat_node_ids[subcat_name] = subcat_id

            # Subcategory is_a Category
            graph.add_edge(subcat_id, cat_id, EdgeType.IS_A)

            # Create entity nodes under this subcategory
            for entity_name in subcat_data.get("entities", []):
                entity_def = entity_defs.get(entity_name, {})
                entity_desc = entity_def.get("description", entity_name)
                cat_path = entity_def.get(
                    "category_path", f"{cat_name} > {subcat_name}"
                )

                entity_id = graph.add_node(
                    node_type=NodeType.CONCEPT,
                    content=entity_desc,
                    name=entity_name,
                    taxonomy_path=cat_path,
                    is_entity=True,
                )
                entity_to_node_id[entity_name] = entity_id

                # Entity is_a Subcategory
                graph.add_edge(entity_id, subcat_id, EdgeType.IS_A)

    # Step 2: Optionally load Formula nodes from training data
    if formula_data_path and os.path.exists(formula_data_path):
        _load_formula_nodes(graph, formula_data_path, entity_to_node_id)

    print(
        f"Ontology initialized: {len(entity_to_node_id)} entity concepts, "
        f"{len(cat_node_ids)} categories"
    )
    return entity_to_node_id


def _load_formula_nodes(
    graph: KnowledgeGraph,
    formula_data_path: str,
    entity_to_node_id: Dict[str, str],
) -> None:
    """
    Extract formula information from Formula training data and create
    Formula nodes with depends_on edges.

    Args:
        graph: KnowledgeGraph to populate.
        formula_data_path: Path to formula training JSONL.
        entity_to_node_id: Mapping from entity names to node IDs.
    """
    seen_formulas = set()

    with open(formula_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            context = item.get("context", "")

            # Extract formula name and expression
            formula_name = _extract_formula_name(context)
            formula_expr = _extract_formula_expression(context)

            if formula_name and formula_name not in seen_formulas:
                seen_formulas.add(formula_name)

                # Create Formula node
                fid = graph.add_node(
                    node_type=NodeType.FORMULA,
                    content=f"{formula_name}: {formula_expr}" if formula_expr else formula_name,
                    name=formula_name,
                    expression=formula_expr or "",
                    from_ontology=True,
                )

                # Extract input variables from formula expression
                if formula_expr:
                    _link_formula_dependencies(
                        graph, fid, formula_expr, entity_to_node_id
                    )

    print(f"  Loaded {len(seen_formulas)} formula nodes from training data")


def _extract_formula_name(context: str) -> Optional[str]:
    """Extract formula name from context string."""
    # Pattern: "Use formula [NAME] to answer..."
    if "Use formula " in context:
        parts = context.split("Use formula ", 1)
        if len(parts) > 1:
            name_part = parts[1].split(" to answer", 1)[0]
            return name_part.strip()
    return None


def _extract_formula_expression(context: str) -> Optional[str]:
    """Extract formula expression from context string."""
    # Pattern: "Formula: [EXPRESSION],"
    if "Formula: " in context:
        parts = context.split("Formula: ", 1)
        if len(parts) > 1:
            expr_part = parts[1].split(", Explanation:", 1)[0]
            expr_part = expr_part.split(",Explanation:", 1)[0]
            return expr_part.strip()
    return None


def _link_formula_dependencies(
    graph: KnowledgeGraph,
    formula_id: str,
    expression: str,
    entity_to_node_id: Dict[str, str],
) -> None:
    """Link a formula to its dependent entity concepts."""
    # Check each known entity name against the formula expression
    expr_lower = expression.lower().replace("_", " ").replace("-", " ")
    for entity_name, entity_nid in entity_to_node_id.items():
        # Convert camelCase entity name to lowercase words for matching
        entity_words = _camel_to_words(entity_name).lower()
        if entity_words in expr_lower or entity_name.lower() in expr_lower:
            graph.add_edge(formula_id, entity_nid, EdgeType.DEPENDS_ON)


def _camel_to_words(name: str) -> str:
    """Convert CamelCase to space-separated words."""
    import re
    words = re.sub(r"([A-Z])", r" \1", name).strip()
    return words


def get_entity_name_to_node_map(graph: KnowledgeGraph) -> Dict[str, str]:
    """
    Build a mapping from entity names to their node IDs in the graph.

    Useful for the GraphConstructor to anchor new strategies/anti-patterns
    to existing concept nodes.

    Args:
        graph: A populated KnowledgeGraph.

    Returns:
        Dict mapping entity names (str) to node IDs (str).
    """
    result = {}
    for nid, data in graph.graph.nodes(data=True):
        if data.get("type") == NodeType.CONCEPT.value and data.get("is_entity"):
            name = data.get("name", "")
            if name:
                result[name] = nid
    return result
