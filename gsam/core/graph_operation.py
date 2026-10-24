"""GraphOperation: a single typed graph mutation produced by the GraphConstructor."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GraphOperation:
    """A single graph operation to apply."""
    op_type: str  # ADD_NODE, ADD_EDGE, UPDATE_ATTR
    node_type: Optional[str] = None
    content: Optional[str] = None
    node_id: Optional[str] = None
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    edge_type: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
