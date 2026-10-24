"""Subgraph-to-text serialization for KnowledgeGraph, split out as a mixin."""

import json
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

from .graph_types import NodeType, EdgeType
from log_config import get_logger

logger = get_logger("graph_memory")


class SerializationMixin:

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

        # --- Conflicts between strategies ---
        conflict_lines = []
        seen_conflict_pairs: set = set()
        for nid, data in subgraph.nodes(data=True):
            if data.get("type") != NodeType.STRATEGY.value:
                continue
            for _, tgt, edata in subgraph.out_edges(nid, data=True):
                if edata.get("type") != EdgeType.CONFLICTS_WITH.value:
                    continue
                pair_key = tuple(sorted([nid, tgt]))
                if pair_key in seen_conflict_pairs:
                    continue
                seen_conflict_pairs.add(pair_key)
                tgt_data = subgraph.nodes.get(tgt, {})
                conflict_lines.append(
                    f"  [{nid}] {data.get('content', '')[:80]}"
                    f" <-> [{tgt}] {tgt_data.get('content', '')[:80]}"
                )

        if conflict_lines:
            lines.append("")
            lines.append("CONFLICTS WITH (mutually exclusive strategies):")
            lines.extend(conflict_lines)

        return "\n".join(lines)
