"""Node and edge type enums for the GSAM knowledge graph."""

from enum import Enum


class NodeType(str, Enum):
    """Types of nodes in the GSAM knowledge graph."""
    STRATEGY = "Strategy"
    ANTI_PATTERN = "AntiPattern"
    CONCEPT = "Concept"
    FORMULA = "Formula"
    CONFUSION = "Confusion"


class EdgeType(str, Enum):
    """Types of edges in the GSAM knowledge graph."""
    IS_A = "is_a"                   # Taxonomic: child -> parent
    PART_OF = "part_of"             # Compositional: component -> aggregate
    DEPENDS_ON = "depends_on"       # Computational: formula -> required entity
    APPLIES_TO = "applies_to"       # Strategy -> Concept it addresses
    FAILS_FOR = "fails_for"         # AntiPattern -> Concept where it fails
    FIXES = "fixes"                 # Strategy -> AntiPattern it resolves
    CONFUSED_WITH = "confused_with" # Concept <-> Concept (bidirectional)
    CONFLICTS_WITH = "conflicts_with"  # Strategy <-> Strategy (mutually exclusive)
