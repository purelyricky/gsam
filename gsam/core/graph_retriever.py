"""
GSAM Graph Retriever

Implements the three-stage ontology-aware retrieval process:
1. Concept Identification: Match query text to Concept nodes
2. Graph Traversal: BFS from matched concepts (depth <= 2)
3. Taxonomic Expansion: Walk is_a edges to include sibling strategies
"""

from typing import Dict, List, Optional, Set, Tuple, Any

from ..graph_memory import KnowledgeGraph, NodeType, EdgeType


class GraphRetriever:
    """
    Ontology-aware subgraph retriever for GSAM.

    Combines concept matching (embedding + keyword) with graph BFS
    traversal to retrieve structurally relevant knowledge.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        retrieval_depth: int = 2,
        max_concepts: int = 10,
        concept_threshold: float = 0.3,
        max_transfer_depth: int = 2,
        embedding_only: bool = False,
    ):
        """
        Initialize the GraphRetriever.

        Args:
            graph: KnowledgeGraph to retrieve from.
            retrieval_depth: BFS depth for graph traversal.
            max_concepts: Maximum concepts to match per query.
            concept_threshold: Embedding similarity threshold for concepts.
            max_transfer_depth: Max taxonomy depth for transfer candidates.
            embedding_only: If True, skip graph traversal (ablation).
        """
        self.graph = graph
        self.retrieval_depth = retrieval_depth
        self.max_concepts = max_concepts
        self.concept_threshold = concept_threshold
        self.max_transfer_depth = max_transfer_depth
        self.embedding_only = embedding_only

    def retrieve(
        self,
        query: str,
        context: str = "",
        top_k: int = 10,
    ) -> Tuple[str, List[str]]:
        """
        Retrieve relevant knowledge as serialized subgraph text.

        Args:
            query: The question text.
            context: Additional context (financial document excerpt).
            top_k: Maximum items to retrieve.

        Returns:
            Tuple of (serialized_context_string, list_of_retrieved_node_ids).
        """
        combined_text = f"{query} {context}".strip()

        if self.embedding_only:
            return self._embedding_only_retrieve(combined_text, top_k)

        # Stage 1: Concept Identification
        matched_concepts = self._identify_concepts(combined_text)

        if not matched_concepts:
            return "(No relevant graph context available)", []

        # Stage 2: Graph Traversal (BFS from concepts)
        seed_ids = [cid for cid, _ in matched_concepts]
        # BFS must NOT traverse is_a / part_of (ontological hierarchy) edges.
        # Traversing them pulls in every sibling/child concept node, flooding
        # the context budget with taxonomy definitions and leaving zero slots
        # for strategies and anti-patterns.  Taxonomic expansion is handled
        # explicitly and deliberately in Stage 3.
        _EXPERIENTIAL_EDGES = {
            EdgeType.APPLIES_TO.value,
            EdgeType.FAILS_FOR.value,
            EdgeType.FIXES.value,
            EdgeType.CONFUSED_WITH.value,
            EdgeType.CONFLICTS_WITH.value,
            EdgeType.DEPENDS_ON.value,
        }
        subgraph = self.graph.get_subgraph(
            seed_ids,
            depth=self.retrieval_depth,
            edge_types=_EXPERIENTIAL_EDGES,
        )

        # Stage 2b: Include nodes reachable via conflicts_with from any
        # strategy or anti-pattern already in the subgraph so the model
        # can see which strategies are mutually exclusive.
        conflicts_extra: Set[str] = set()
        for nid in list(subgraph.nodes()):
            ntype = subgraph.nodes[nid].get("type", "")
            if ntype not in (NodeType.STRATEGY.value, NodeType.ANTI_PATTERN.value):
                continue
            for _, nbr, edata in self.graph.graph.out_edges(nid, data=True):
                if edata.get("type") == EdgeType.CONFLICTS_WITH.value and nbr not in subgraph.nodes:
                    conflicts_extra.add(nbr)
            for nbr, _, edata in self.graph.graph.in_edges(nid, data=True):
                if edata.get("type") == EdgeType.CONFLICTS_WITH.value and nbr not in subgraph.nodes:
                    conflicts_extra.add(nbr)

        if conflicts_extra:
            all_ids_stage2 = set(subgraph.nodes()) | conflicts_extra
            subgraph = self.graph.get_subgraph(list(all_ids_stage2), depth=0)

        # Stage 3: Taxonomic Expansion
        transfer_nodes = self._taxonomic_expansion(matched_concepts, subgraph)

        # Add transfer nodes to subgraph
        if transfer_nodes:
            all_ids = set(subgraph.nodes()) | set(transfer_nodes.keys())
            expanded_subgraph = self.graph.get_subgraph(list(all_ids), depth=0)
            # Mark transferred nodes
            for nid in transfer_nodes:
                if nid in expanded_subgraph.nodes:
                    expanded_subgraph.nodes[nid]["tentative"] = True
            subgraph = expanded_subgraph

        # Collect failure cascade warnings
        self._annotate_failure_cascades(subgraph, seed_ids)

        # Cap retrieved nodes using separate budgets for concepts vs. knowledge
        # nodes (strategies, anti-patterns, formulas, confusions).
        # Using a single budget with concepts at float("inf") priority means
        # that whenever >30 concept nodes appear (common with the full XBRL
        # ontology), all slots go to taxonomy definitions and zero slots remain
        # for actionable strategies/anti-patterns — the model then performs
        # worse than the base LLM.
        MAX_CONCEPT_NODES = 10
        MAX_KNOWLEDGE_NODES = 20
        all_ids = list(subgraph.nodes())
        concept_ids = [
            nid for nid in all_ids
            if self.graph.graph.nodes.get(nid, {}).get("type") == NodeType.CONCEPT.value
        ]
        knowledge_ids = [nid for nid in all_ids if nid not in set(concept_ids)]

        if len(concept_ids) > MAX_CONCEPT_NODES:
            concept_ids = concept_ids[:MAX_CONCEPT_NODES]

        if len(knowledge_ids) > MAX_KNOWLEDGE_NODES:
            def _priority(nid):
                data = self.graph.graph.nodes.get(nid, {})
                return data.get("helpful_count", 0) * data.get("confidence", 1.0)
            knowledge_ids = sorted(knowledge_ids, key=_priority, reverse=True)[:MAX_KNOWLEDGE_NODES]

        trimmed = set(concept_ids) | set(knowledge_ids)
        if len(trimmed) < len(all_ids):
            subgraph = self.graph.get_subgraph(list(trimmed), depth=0)

        # Serialize and return
        serialized = self.graph.serialize_subgraph(subgraph)
        retrieved_ids = list(subgraph.nodes())

        return serialized, retrieved_ids

    def _identify_concepts(
        self, text: str
    ) -> List[Tuple[str, float]]:
        """
        Stage 1: Identify relevant concept nodes.

        Combines keyword matching and embedding similarity.

        Args:
            text: Combined query + context text.

        Returns:
            List of (concept_node_id, score) tuples, sorted by score.
        """
        matches = {}

        # Keyword matching (exact concept names in text)
        keyword_matches = self.graph.find_concepts_by_name(text)
        for cid in keyword_matches:
            matches[cid] = 1.0  # Perfect score for keyword matches

        # Embedding similarity matching
        embedding_matches = self.graph.find_concepts_by_text(
            text,
            top_k=self.max_concepts,
            threshold=self.concept_threshold,
        )
        for cid, score in embedding_matches:
            if cid in matches:
                matches[cid] = max(matches[cid], score)
            else:
                matches[cid] = score

        # Sort by score descending, limit to max_concepts
        sorted_matches = sorted(
            matches.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_matches[:self.max_concepts]

    def _taxonomic_expansion(
        self,
        matched_concepts: List[Tuple[str, float]],
        current_subgraph,
    ) -> Dict[str, str]:
        """
        Stage 3: Expand retrieval via taxonomic (is_a) edges.

        For each matched concept, find strategies from sibling concepts
        (those sharing a direct parent in the taxonomy).

        Args:
            matched_concepts: List of (concept_id, score) from Stage 1.
            current_subgraph: Current subgraph from Stage 2.

        Returns:
            Dict mapping transfer_node_id -> source_concept_id.
        """
        transfer_nodes = {}

        for cid, score in matched_concepts:
            # Get siblings (concepts sharing a parent via is_a)
            siblings = self.graph.get_siblings(cid, edge_type=EdgeType.IS_A.value)

            for sibling_id in siblings:
                if sibling_id in current_subgraph.nodes:
                    continue

                # Check LCA depth
                lca_depth = self.graph.lowest_common_ancestor_depth(
                    cid, sibling_id, edge_type=EdgeType.IS_A.value
                )
                if lca_depth < 0 or lca_depth > self.max_transfer_depth:
                    continue

                # Find strategies applied to this sibling
                for nbr, _, edata in self.graph.graph.in_edges(
                    sibling_id, data=True
                ):
                    if edata.get("type") == EdgeType.APPLIES_TO.value:
                        nbr_data = self.graph.graph.nodes.get(nbr, {})
                        if nbr_data.get("type") == NodeType.STRATEGY.value:
                            if nbr not in current_subgraph.nodes:
                                transfer_nodes[nbr] = sibling_id

        return transfer_nodes

    def _annotate_failure_cascades(
        self, subgraph, concept_ids: List[str]
    ) -> None:
        """
        Find anti-patterns that cascade through formula dependencies.

        For each concept in the subgraph, check if there are anti-patterns
        that affect formulas depending on that concept.
        """
        for cid in concept_ids:
            if cid not in subgraph.nodes:
                continue
            # Find anti-patterns failing for this concept
            for ap_id, _, edata in self.graph.graph.in_edges(cid, data=True):
                ap_data = self.graph.graph.nodes.get(ap_id, {})
                if (edata.get("type") == EdgeType.FAILS_FOR.value
                        and ap_data.get("type") == NodeType.ANTI_PATTERN.value):
                    # Add anti-pattern to subgraph if not already there
                    if ap_id not in subgraph.nodes:
                        subgraph.add_node(ap_id, **ap_data)
                        subgraph.add_edge(ap_id, cid, **edata)

                    # Find formulas depending on this concept
                    for fid, _, fedata in self.graph.graph.in_edges(cid, data=True):
                        fdata = self.graph.graph.nodes.get(fid, {})
                        if (fedata.get("type") == EdgeType.DEPENDS_ON.value
                                and fdata.get("type") == NodeType.FORMULA.value):
                            if fid not in subgraph.nodes:
                                subgraph.add_node(fid, **fdata)
                                subgraph.add_edge(fid, cid, **fedata)

    def _embedding_only_retrieve(
        self, text: str, top_k: int
    ) -> Tuple[str, List[str]]:
        """
        Ablation: embedding-only retrieval without graph traversal.

        Retrieves top-k strategies and anti-patterns by cosine similarity.
        """
        from ..graph_memory import EMBEDDINGS_AVAILABLE
        if not EMBEDDINGS_AVAILABLE:
            return "(No embedding model available)", []

        emb_model = self.graph.embedding_model
        if emb_model is None:
            return "(No embedding model available)", []

        import numpy as np
        query_emb = emb_model.encode(text, convert_to_numpy=True)
        query_emb = query_emb.ravel()
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)

        scores = []
        for nid, data in self.graph.graph.nodes(data=True):
            ntype = data.get("type", "")
            if ntype not in (NodeType.STRATEGY.value, NodeType.ANTI_PATTERN.value):
                continue
            if nid not in self.graph.embeddings:
                continue
            sim = float(np.dot(query_emb, self.graph.embeddings[nid].ravel()))
            scores.append((nid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scores[:top_k]

        if not top_nodes:
            return "(No relevant strategies found)", []

        lines = ["RETRIEVED STRATEGIES (by embedding similarity):"]
        for nid, sim in top_nodes:
            data = self.graph.graph.nodes[nid]
            ntype = data.get("type", "")
            content = data.get("content", "")
            h = data.get("helpful_count", 0)
            n = data.get("harmful_count", 0)
            lines.append(
                f"  [{nid}] ({ntype}) {content}"
                f" (helpful={h}, harmful={n}, similarity={sim:.2f})"
            )

        return "\n".join(lines), [nid for nid, _ in top_nodes]

    def get_risks_for_formula(self, formula_id: str) -> List[Dict[str, Any]]:
        """
        Check for upstream anti-patterns that affect a formula.

        Implements: risks(f) = {a : exists c s.t. (f, depends_on, c) AND (a, fails_for, c)}

        Args:
            formula_id: Node ID of a formula node.

        Returns:
            List of risk descriptions.
        """
        risks = []

        # Get concepts this formula depends on
        dep_concepts = []
        for _, dep_id, edata in self.graph.graph.out_edges(formula_id, data=True):
            if edata.get("type") == EdgeType.DEPENDS_ON.value:
                dep_concepts.append(dep_id)

        # Check each dependent concept for anti-patterns
        for cid in dep_concepts:
            for ap_id, _, edata in self.graph.graph.in_edges(cid, data=True):
                if edata.get("type") == EdgeType.FAILS_FOR.value:
                    ap_data = self.graph.graph.nodes.get(ap_id, {})
                    if ap_data.get("type") == NodeType.ANTI_PATTERN.value:
                        c_data = self.graph.graph.nodes.get(cid, {})
                        risks.append({
                            "anti_pattern_id": ap_id,
                            "anti_pattern": ap_data.get("content", ""),
                            "affected_concept_id": cid,
                            "affected_concept": c_data.get("name", ""),
                            "severity": ap_data.get("severity", "medium"),
                        })

        return risks
