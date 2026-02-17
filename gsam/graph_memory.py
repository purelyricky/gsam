"""
Graph-Structured Adaptive Memory (GSAM) - Knowledge Graph Module

Replaces ACE's flat bullet-point storage (playbook_utils.py) with a
typed, attributed knowledge graph using NetworkX.

Node types: Strategy, AntiPattern, Concept, Formula, Confusion
Edge types: is_a, part_of, depends_on, applies_to, fails_for, fixes,
            confused_with, conflicts_with
"""

import json
import time
import uuid
import numpy as np
import networkx as nx
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available. "
          "Install with: pip install sentence-transformers")


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


class KnowledgeGraph:
    """
    Ontology-grounded knowledge graph for GSAM.

    Replaces flat bullet storage with typed nodes and edges.
    Uses NetworkX DiGraph internally with typed attributes.
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize an empty knowledge graph.

        Args:
            embedding_model_name: Sentence transformer model for similarity.
        """
        self.graph = nx.DiGraph()
        self.embeddings: Dict[str, np.ndarray] = {}
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None
        self._next_ids: Dict[str, int] = defaultdict(int)
        self.created_at = time.time()
        self.tasks_processed = 0

        # Ablation flags
        self.use_typed_edges = True

    @property
    def embedding_model(self):
        """Lazy-load embedding model."""
        if self._embedding_model is None and EMBEDDINGS_AVAILABLE:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    # ------------------------------------------------------------------
    # Node Operations
    # ------------------------------------------------------------------

    def _generate_id(self, node_type: NodeType) -> str:
        """Generate a unique node ID with type prefix."""
        prefix_map = {
            NodeType.STRATEGY: "S",
            NodeType.ANTI_PATTERN: "A",
            NodeType.CONCEPT: "C",
            NodeType.FORMULA: "F",
            NodeType.CONFUSION: "X",
        }
        prefix = prefix_map.get(node_type, "N")
        self._next_ids[prefix] += 1
        return f"{prefix}:{self._next_ids[prefix]:04d}"

    def add_node(
        self,
        node_type: NodeType,
        content: str,
        node_id: Optional[str] = None,
        **attrs
    ) -> str:
        """
        Add a node to the knowledge graph.

        Args:
            node_type: Type of the node (Strategy, AntiPattern, etc.)
            content: Natural language content of the node.
            node_id: Optional explicit ID. Auto-generated if None.
            **attrs: Additional attributes (helpful_count, harmful_count, etc.)

        Returns:
            The node ID of the added node.
        """
        if node_id is None:
            node_id = self._generate_id(node_type)

        # Update the counter if an explicit ID is used with the same prefix
        if node_id and ":" in node_id:
            prefix, num_str = node_id.split(":", 1)
            try:
                num = int(num_str)
                self._next_ids[prefix] = max(self._next_ids[prefix], num)
            except ValueError:
                pass

        default_attrs = {
            "type": node_type.value if isinstance(node_type, NodeType) else node_type,
            "content": content,
            "helpful_count": 0,
            "harmful_count": 0,
            "confidence": 1.0,
            "created_at": time.time(),
            "last_used": time.time(),
            "use_count": 0,
        }
        default_attrs.update(attrs)
        self.graph.add_node(node_id, **default_attrs)

        # Compute and cache embedding
        if EMBEDDINGS_AVAILABLE and content:
            self._update_embedding(node_id, content)

        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        **attrs
    ) -> None:
        """
        Add a typed edge between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Type of the edge.
            **attrs: Additional edge attributes.
        """
        if source_id not in self.graph:
            raise ValueError(f"Source node {source_id} not in graph")
        if target_id not in self.graph:
            raise ValueError(f"Target node {target_id} not in graph")

        actual_type = edge_type.value if isinstance(edge_type, EdgeType) else edge_type
        if not self.use_typed_edges:
            actual_type = "related_to"

        edge_attrs = {
            "type": actual_type,
            "weight": 1.0,
            "created_at": time.time(),
        }
        edge_attrs.update(attrs)
        self.graph.add_edge(source_id, target_id, **edge_attrs)

    def merge_nodes(self, existing_id: str, new_content: str, **new_attrs) -> None:
        """
        Merge new information into an existing node.

        Combines content and sums counters.

        Args:
            existing_id: ID of the existing node to merge into.
            new_content: New content to merge.
            **new_attrs: Additional attributes to update.
        """
        if existing_id not in self.graph:
            raise ValueError(f"Node {existing_id} not in graph")

        node = self.graph.nodes[existing_id]
        old_content = node.get("content", "")

        # Merge content: append if substantially different
        if new_content and new_content not in old_content:
            node["content"] = f"{old_content} | {new_content}"

        # Sum counters
        for key in ["helpful_count", "harmful_count", "use_count"]:
            if key in new_attrs:
                node[key] = node.get(key, 0) + new_attrs.pop(key)

        # Update confidence as weighted average
        if "confidence" in new_attrs:
            old_conf = node.get("confidence", 1.0)
            new_conf = new_attrs.pop("confidence")
            node["confidence"] = (old_conf + new_conf) / 2.0

        # Update last_used
        node["last_used"] = time.time()

        # Apply remaining attrs
        node.update(new_attrs)

        # Update embedding
        if EMBEDDINGS_AVAILABLE:
            self._update_embedding(existing_id, node["content"])

    def update_node_attr(self, node_id: str, **attrs) -> None:
        """Update attributes of an existing node."""
        if node_id not in self.graph:
            return
        self.graph.nodes[node_id].update(attrs)

    def find_similar_node(
        self,
        content: str,
        node_type: NodeType,
        threshold: float = 0.9,
    ) -> Optional[str]:
        """
        Find an existing node of the same type with similar content.

        Args:
            content: Content to compare against.
            node_type: Required node type to match.
            threshold: Cosine similarity threshold.

        Returns:
            Node ID of the most similar node, or None.
        """
        if not EMBEDDINGS_AVAILABLE or not content:
            return None

        target_type = node_type.value if isinstance(node_type, NodeType) else node_type
        query_emb = self.embedding_model.encode(content, convert_to_numpy=True)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)

        best_id = None
        best_sim = -1.0

        for nid, data in self.graph.nodes(data=True):
            if data.get("type") != target_type:
                continue
            if nid not in self.embeddings:
                continue
            sim = float(np.dot(query_emb, self.embeddings[nid]))
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best_id = nid

        return best_id

    # ------------------------------------------------------------------
    # Graph Retrieval
    # ------------------------------------------------------------------

    def get_subgraph(
        self,
        seed_ids: List[str],
        depth: int = 2,
        edge_types: Optional[Set[str]] = None,
    ) -> nx.DiGraph:
        """
        Extract a subgraph via BFS from seed nodes up to given depth.

        Args:
            seed_ids: Starting node IDs.
            depth: Maximum BFS depth.
            edge_types: If set, only traverse edges of these types.

        Returns:
            A NetworkX DiGraph subgraph.
        """
        visited = set()
        frontier = set(nid for nid in seed_ids if nid in self.graph)
        all_nodes = set(frontier)

        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                # Outgoing edges
                for _, neighbor, edata in self.graph.out_edges(nid, data=True):
                    if edge_types and edata.get("type") not in edge_types:
                        continue
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        all_nodes.add(neighbor)
                # Incoming edges
                for neighbor, _, edata in self.graph.in_edges(nid, data=True):
                    if edge_types and edata.get("type") not in edge_types:
                        continue
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        all_nodes.add(neighbor)
            visited.update(frontier)
            frontier = next_frontier - visited

        return self.graph.subgraph(all_nodes).copy()

    def get_nodes_by_type(self, node_type: NodeType) -> List[str]:
        """Return all node IDs of a given type."""
        target = node_type.value if isinstance(node_type, NodeType) else node_type
        return [
            nid for nid, data in self.graph.nodes(data=True)
            if data.get("type") == target
        ]

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "both",
    ) -> List[Tuple[str, Dict]]:
        """
        Get neighbors of a node, optionally filtered by edge type.

        Args:
            node_id: Source node.
            edge_type: Optional filter.
            direction: 'out', 'in', or 'both'.

        Returns:
            List of (neighbor_id, edge_data) tuples.
        """
        results = []
        et = edge_type.value if isinstance(edge_type, EdgeType) else edge_type

        if direction in ("out", "both"):
            for _, nbr, edata in self.graph.out_edges(node_id, data=True):
                if et is None or edata.get("type") == et:
                    results.append((nbr, edata))

        if direction in ("in", "both"):
            for nbr, _, edata in self.graph.in_edges(node_id, data=True):
                if et is None or edata.get("type") == et:
                    results.append((nbr, edata))

        return results

    def find_concepts_by_text(
        self,
        text: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """
        Find concept nodes matching query text via embedding similarity.

        Args:
            text: Query text.
            top_k: Maximum concepts to return.
            threshold: Minimum similarity score.

        Returns:
            List of (concept_node_id, similarity_score) sorted descending.
        """
        if not EMBEDDINGS_AVAILABLE or not text:
            return []

        query_emb = self.embedding_model.encode(text, convert_to_numpy=True)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)

        concept_ids = self.get_nodes_by_type(NodeType.CONCEPT)
        scores = []

        for cid in concept_ids:
            if cid not in self.embeddings:
                continue
            sim = float(np.dot(query_emb, self.embeddings[cid]))
            if sim >= threshold:
                scores.append((cid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def find_concepts_by_name(self, text: str) -> List[str]:
        """
        Find concept nodes by keyword matching against concept names.

        Args:
            text: Text that may contain concept names.

        Returns:
            List of matching concept node IDs.
        """
        text_lower = text.lower()
        matches = []
        for nid in self.get_nodes_by_type(NodeType.CONCEPT):
            name = self.graph.nodes[nid].get("name", "").lower()
            content = self.graph.nodes[nid].get("content", "").lower()
            # Check if concept name appears in query text
            if name and name in text_lower:
                matches.append(nid)
            elif content and len(content) > 3 and content in text_lower:
                matches.append(nid)
        return matches

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(
        self,
        min_degree: int = 2,
        min_confidence: float = 0.1,
        max_age_epochs: int = 10,
    ) -> int:
        """
        Remove low-utility nodes from the graph.

        Nodes are removed if they have low degree, low confidence,
        and haven't been used recently. Concept nodes from the ontology
        are never pruned.

        Args:
            min_degree: Minimum degree to survive pruning.
            min_confidence: Minimum confidence to survive pruning.
            max_age_epochs: Max tasks since last use before eligible.

        Returns:
            Number of nodes removed.
        """
        to_remove = []
        now = time.time()

        for nid, data in self.graph.nodes(data=True):
            # Never prune Concept nodes (ontology backbone)
            if data.get("type") == NodeType.CONCEPT.value:
                continue
            # Never prune Formula nodes initialized from ontology
            if data.get("type") == NodeType.FORMULA.value and data.get("from_ontology"):
                continue

            degree = self.graph.degree(nid)
            confidence = data.get("confidence", 1.0)
            use_count = data.get("use_count", 0)

            if degree < min_degree and confidence < min_confidence and use_count == 0:
                to_remove.append(nid)

        for nid in to_remove:
            if nid in self.embeddings:
                del self.embeddings[nid]
            self.graph.remove_node(nid)

        return len(to_remove)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize_subgraph(self, subgraph: nx.DiGraph) -> str:
        """
        Serialize a subgraph into structured natural language for the Generator.

        Args:
            subgraph: A NetworkX subgraph to serialize.

        Returns:
            Structured text representation of the subgraph.
        """
        if len(subgraph) == 0:
            return "(No relevant graph context available)"

        # Organize nodes by type
        concepts = []
        strategies = []
        anti_patterns = []
        formulas = []
        confusions = []

        for nid, data in subgraph.nodes(data=True):
            ntype = data.get("type", "")
            entry = {"id": nid, **data}
            if ntype == NodeType.CONCEPT.value:
                concepts.append(entry)
            elif ntype == NodeType.STRATEGY.value:
                strategies.append(entry)
            elif ntype == NodeType.ANTI_PATTERN.value:
                anti_patterns.append(entry)
            elif ntype == NodeType.FORMULA.value:
                formulas.append(entry)
            elif ntype == NodeType.CONFUSION.value:
                confusions.append(entry)

        lines = []

        # --- Concepts with their strategies and anti-patterns ---
        if concepts:
            lines.append("RELEVANT CONCEPTS:")
            for c in concepts:
                cid = c["id"]
                taxonomy_path = c.get("taxonomy_path", "")
                path_str = f" ({taxonomy_path})" if taxonomy_path else ""
                lines.append(f"  [{cid}] {c.get('name', c.get('content', ''))}{path_str}")

                # Find strategies applying to this concept
                c_strats = []
                for sid, _, edata in subgraph.in_edges(cid, data=True):
                    if edata.get("type") == EdgeType.APPLIES_TO.value:
                        snode = subgraph.nodes.get(sid, {})
                        if snode.get("type") == NodeType.STRATEGY.value:
                            conf = "high" if snode.get("confidence", 0) > 0.7 else "medium"
                            tentative = " (tentative - transferred)" if snode.get("tentative") else ""
                            lines.append(
                                f"    Strategy [{sid}]: {snode.get('content', '')}"
                                f" (helpful={snode.get('helpful_count', 0)},"
                                f" harmful={snode.get('harmful_count', 0)},"
                                f" confidence={conf}){tentative}"
                            )

                # Find anti-patterns for this concept
                for sid, _, edata in subgraph.in_edges(cid, data=True):
                    if edata.get("type") == EdgeType.FAILS_FOR.value:
                        anode = subgraph.nodes.get(sid, {})
                        if anode.get("type") == NodeType.ANTI_PATTERN.value:
                            root_cause = anode.get("root_cause", "")
                            rc_str = f" (root_cause: {root_cause})" if root_cause else ""
                            severity = anode.get("severity", "medium")
                            lines.append(
                                f"    Anti-pattern [{sid}]: DO NOT {anode.get('content', '')}"
                                f"{rc_str} (severity={severity},"
                                f" occurrences={anode.get('harmful_count', 0)})"
                            )

        # --- Failure Warnings (cascading) ---
        if anti_patterns:
            cascade_warnings = []
            for ap in anti_patterns:
                ap_id = ap["id"]
                # Find formulas affected by this anti-pattern's target concepts
                for _, target, edata in subgraph.out_edges(ap_id, data=True):
                    if edata.get("type") == EdgeType.FAILS_FOR.value:
                        target_data = subgraph.nodes.get(target, {})
                        # Check if any formula depends on this concept
                        for fid, _, fedata in subgraph.in_edges(target, data=True):
                            fnode = subgraph.nodes.get(fid, {})
                            if fnode.get("type") == NodeType.FORMULA.value:
                                cascade_warnings.append(
                                    f"  [{ap_id}] affects Formula [{fid}]"
                                    f" {fnode.get('content', '')}"
                                    f" -- If {target_data.get('name', target)} is wrong,"
                                    f" this formula result will also be wrong"
                                )

            if cascade_warnings:
                lines.append("")
                lines.append("FAILURE WARNINGS (cascading):")
                lines.extend(cascade_warnings)

        # --- Formulas ---
        if formulas:
            lines.append("")
            lines.append("FORMULA DEPENDENCIES:")
            for f in formulas:
                fid = f["id"]
                expr = f.get("expression", f.get("content", ""))
                lines.append(f"  [{fid}] {expr}")
                # List dependencies
                for _, dep, edata in subgraph.out_edges(fid, data=True):
                    if edata.get("type") == EdgeType.DEPENDS_ON.value:
                        dep_data = subgraph.nodes.get(dep, {})
                        lines.append(
                            f"    depends_on: [{dep}] {dep_data.get('name', dep_data.get('content', ''))}"
                        )

        # --- Confusions ---
        if confusions:
            lines.append("")
            lines.append("KNOWN CONFUSIONS:")
            for x in confusions:
                lines.append(
                    f"  [{x['id']}] {x.get('content', '')}"
                    f" -- Distinguishing criteria: {x.get('distinguishing_criteria', 'N/A')}"
                )

        # --- Strategies not yet shown under concepts ---
        orphan_strats = []
        shown_strat_ids = set()
        for c in concepts:
            for sid, _, edata in subgraph.in_edges(c["id"], data=True):
                if edata.get("type") == EdgeType.APPLIES_TO.value:
                    shown_strat_ids.add(sid)
        for s in strategies:
            if s["id"] not in shown_strat_ids:
                tentative = " (tentative - transferred)" if s.get("tentative") else ""
                orphan_strats.append(
                    f"  [{s['id']}] {s.get('content', '')}"
                    f" (helpful={s.get('helpful_count', 0)},"
                    f" harmful={s.get('harmful_count', 0)}){tentative}"
                )
        if orphan_strats:
            lines.append("")
            lines.append("ADDITIONAL STRATEGIES:")
            lines.extend(orphan_strats)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence (JSON)
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """Serialize the entire graph to a JSON-serializable dict."""
        nodes = []
        for nid, data in self.graph.nodes(data=True):
            node_data = {"id": nid}
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool, list, dict)):
                    node_data[k] = v
                else:
                    node_data[k] = str(v)
            nodes.append(node_data)

        edges = []
        for src, tgt, data in self.graph.edges(data=True):
            edge_data = {"source": src, "target": tgt}
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool, list, dict)):
                    edge_data[k] = v
                else:
                    edge_data[k] = str(v)
            edges.append(edge_data)

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "created_at": self.created_at,
                "tasks_processed": self.tasks_processed,
                "next_ids": dict(self._next_ids),
                "embedding_model": self.embedding_model_name,
                "use_typed_edges": self.use_typed_edges,
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> "KnowledgeGraph":
        """Load a KnowledgeGraph from a JSON dict."""
        meta = data.get("metadata", {})
        kg = cls(embedding_model_name=meta.get("embedding_model", "all-MiniLM-L6-v2"))
        kg.created_at = meta.get("created_at", time.time())
        kg.tasks_processed = meta.get("tasks_processed", 0)
        kg._next_ids = defaultdict(int, meta.get("next_ids", {}))
        kg.use_typed_edges = meta.get("use_typed_edges", True)

        for node in data.get("nodes", []):
            # Copy to avoid mutating the input dict
            node = dict(node)
            nid = node.pop("id")
            kg.graph.add_node(nid, **node)
            # Recompute embedding
            content = node.get("content", "")
            if content and EMBEDDINGS_AVAILABLE:
                kg._update_embedding(nid, content)

        for edge in data.get("edges", []):
            edge = dict(edge)
            src = edge.pop("source")
            tgt = edge.pop("target")
            kg.graph.add_edge(src, tgt, **edge)

        return kg

    def save(self, path: str) -> None:
        """Save graph to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """Load graph from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json(data)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary statistics about the graph."""
        node_counts = defaultdict(int)
        edge_counts = defaultdict(int)

        for _, data in self.graph.nodes(data=True):
            node_counts[data.get("type", "Unknown")] += 1

        for _, _, data in self.graph.edges(data=True):
            edge_counts[data.get("type", "Unknown")] += 1

        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()

        # Concept coverage: fraction of concepts with at least one strategy
        concept_ids = self.get_nodes_by_type(NodeType.CONCEPT)
        concepts_with_strategy = 0
        for cid in concept_ids:
            has_strategy = any(
                self.graph.nodes[nbr].get("type") == NodeType.STRATEGY.value
                for nbr, _, edata in self.graph.in_edges(cid, data=True)
                if edata.get("type") == EdgeType.APPLIES_TO.value
            )
            if has_strategy:
                concepts_with_strategy += 1

        concept_coverage = (
            concepts_with_strategy / len(concept_ids) if concept_ids else 0.0
        )

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_counts": dict(node_counts),
            "edge_counts": dict(edge_counts),
            "density": nx.density(self.graph) if total_nodes > 1 else 0.0,
            "concept_coverage": concept_coverage,
            "concepts_with_strategy": concepts_with_strategy,
            "total_concepts": len(concept_ids),
            "tasks_processed": self.tasks_processed,
        }

    # ------------------------------------------------------------------
    # Multi-Epoch Graph Refinement
    # ------------------------------------------------------------------

    def consolidate_epoch(
        self,
        edge_similarity_threshold: float = 0.85,
    ) -> Dict[str, int]:
        """
        Run multi-epoch graph refinement after each epoch (Paper §5.6).

        Three operations:
        1. Edge Discovery: For each Strategy node, check if it should
           apply_to additional Concepts based on shared neighborhoods.
        2. Edge Weight Reinforcement: Increment weights on applies_to
           edges for strategies confirmed helpful; create/strengthen
           fails_for edges for strategies that failed.
        3. Cross-Node Consolidation: Merge Strategy nodes with high
           cosine similarity AND identical concept neighborhoods.

        Args:
            edge_similarity_threshold: Cosine similarity for merging
                strategy nodes during consolidation.

        Returns:
            Dict with counts of operations performed.
        """
        stats = {"edges_discovered": 0, "edges_reinforced": 0, "nodes_merged": 0}

        # --- 1. Edge Discovery ---
        # For each Strategy, find Concepts it applies_to. Then look at
        # sibling Concepts (via is_a). If the strategy was helpful and
        # the sibling has no strategy yet, create a tentative applies_to.
        strategy_ids = self.get_nodes_by_type(NodeType.STRATEGY)
        for sid in strategy_ids:
            sdata = self.graph.nodes[sid]
            if sdata.get("helpful_count", 0) <= 0:
                continue

            # Get concepts this strategy already applies to
            applied_concepts = set()
            for _, tgt, edata in self.graph.out_edges(sid, data=True):
                if edata.get("type") == EdgeType.APPLIES_TO.value:
                    applied_concepts.add(tgt)

            # For each applied concept, consider siblings
            new_targets = set()
            for cid in list(applied_concepts):
                siblings = self.get_siblings(cid, edge_type=EdgeType.IS_A.value)
                for sib in siblings:
                    if sib not in applied_concepts and sib not in new_targets:
                        # Check if sibling lacks any strategy
                        has_strategy = any(
                            self.graph.nodes.get(src, {}).get("type") == NodeType.STRATEGY.value
                            for src, _, ed in self.graph.in_edges(sib, data=True)
                            if ed.get("type") == EdgeType.APPLIES_TO.value
                        )
                        if not has_strategy:
                            new_targets.add(sib)

            for tgt in new_targets:
                if tgt in self.graph:
                    self.graph.add_edge(
                        sid, tgt,
                        type=EdgeType.APPLIES_TO.value,
                        weight=0.5,
                        tentative=True,
                        created_at=time.time(),
                    )
                    stats["edges_discovered"] += 1

        # --- 2. Edge Weight Reinforcement ---
        for sid in strategy_ids:
            sdata = self.graph.nodes[sid]
            helpful = sdata.get("helpful_count", 0)
            harmful = sdata.get("harmful_count", 0)

            for _, tgt, edata in self.graph.out_edges(sid, data=True):
                if edata.get("type") == EdgeType.APPLIES_TO.value:
                    old_w = edata.get("weight", 1.0)
                    # Reinforce based on net helpfulness
                    delta = helpful - harmful
                    if delta > 0:
                        edata["weight"] = old_w + delta * 0.1
                        stats["edges_reinforced"] += 1

        # --- 3. Cross-Node Consolidation ---
        if not EMBEDDINGS_AVAILABLE:
            return stats

        # Find strategy pairs with high similarity AND same neighborhoods
        merged = set()
        for i, sid_a in enumerate(strategy_ids):
            if sid_a in merged or sid_a not in self.embeddings:
                continue
            for sid_b in strategy_ids[i + 1:]:
                if sid_b in merged or sid_b not in self.embeddings:
                    continue

                sim = float(np.dot(self.embeddings[sid_a], self.embeddings[sid_b]))
                if sim < edge_similarity_threshold:
                    continue

                # Check if neighborhoods match (same connected concepts)
                concepts_a = {
                    tgt for _, tgt, ed in self.graph.out_edges(sid_a, data=True)
                    if ed.get("type") == EdgeType.APPLIES_TO.value
                }
                concepts_b = {
                    tgt for _, tgt, ed in self.graph.out_edges(sid_b, data=True)
                    if ed.get("type") == EdgeType.APPLIES_TO.value
                }

                if concepts_a and concepts_a == concepts_b:
                    # Merge b into a
                    self.merge_nodes(sid_a, self.graph.nodes[sid_b].get("content", ""))
                    # Re-link b's edges to a
                    for _, tgt, ed in list(self.graph.out_edges(sid_b, data=True)):
                        if not self.graph.has_edge(sid_a, tgt):
                            self.graph.add_edge(sid_a, tgt, **ed)
                    for src, _, ed in list(self.graph.in_edges(sid_b, data=True)):
                        if not self.graph.has_edge(src, sid_a):
                            self.graph.add_edge(src, sid_a, **ed)
                    self.graph.remove_node(sid_b)
                    if sid_b in self.embeddings:
                        del self.embeddings[sid_b]
                    merged.add(sid_b)
                    stats["nodes_merged"] += 1

        return stats

    # ------------------------------------------------------------------
    # Taxonomy helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_embedding(self, node_id: str, content: str) -> None:
        """Compute and cache an embedding for a node."""
        if not EMBEDDINGS_AVAILABLE or self.embedding_model is None:
            return
        try:
            emb = self.embedding_model.encode(content, convert_to_numpy=True)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            self.embeddings[node_id] = emb
        except Exception:
            pass

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"KnowledgeGraph(nodes={s['total_nodes']}, edges={s['total_edges']}, "
            f"concepts={s['total_concepts']}, coverage={s['concept_coverage']:.2f})"
        )
