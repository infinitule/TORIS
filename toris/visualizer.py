"""TORIS Visualizer — Converting the Relational Field to interactive HTML.
(Section 9.8 Visualization Addendum)
"""

from __future__ import annotations
import os
from typing import List, Dict, Tuple

try:
    from pyvis.network import Network
except ModuleNotFoundError as _e:  # optional dependency
    Network = None
    _PYVIS_ERR = _e
else:
    _PYVIS_ERR = None

from toris.field.relational_field import RelationalField
from toris.primitives.relator import Relator
from toris.primitives.relation_types import RelationType

# Visual mapping for Relation Types
TYPE_COLORS = {
    RelationType.CAUSAL: "#3498db",      # Blue
    RelationType.CONDITIONAL: "#f1c40f", # Yellow
    RelationType.CONTRADICTS: "#e74c3c", # Red
    RelationType.CONTAINS: "#9b59b6",    # Purple
    RelationType.ENABLES: "#2ecc71",    # Green
    RelationType.VIOLATES: "#e67e22",    # Orange
    RelationType.ANALOGOUS: "#1abc9c",   # Teal
    RelationType.REFINES: "#34495e",     # Dark Blue
    RelationType.TEMPORAL_BEFORE: "#95a5a6", # Grey
    RelationType.EVIDENCES: "#d35400",   # Rust
    RelationType.NEGATES: "#c0392b",     # Dark Red
    RelationType.INSTANTIATES: "#7f8c8d",# Silver
}

DEFAULT_COLOR = "#bdc3c7"

class FieldVisualizer:
    """Generates interactive HTML visualizations of the TORIS Relational Field."""

    def __init__(self, field: RelationalField):
        if Network is None:
            raise ImportError(
                "FieldVisualizer requires the optional 'pyvis' dependency. "
                "Install it with:  pip install 'toris[viz]'  (or  pip install pyvis)"
            ) from _PYVIS_ERR
        self.field = field

    def export_html(self, filename: str, title: str = "TORIS Relational Field"):
        """Export the current field state as an interactive HTML graph."""
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)

        # 1. Add Concepts (Nodes)
        # We use a dummy salience for node size if not provided by a goal manifold
        for concept in self.field.concepts():
            size = 10 + (50 * 0.5) # Baseline size
            net.add_node(
                concept.id,
                label=concept.id,
                size=size,
                color="#ffffff",
                title=f"Concept: {concept.id}"
            )

        # 2. Add Relators (Edges)
        for r in self.field.relators():
            color = TYPE_COLORS.get(r.tau, DEFAULT_COLOR)

            # If surprise is high, make the edge glow/brighter
            if r.epsilon > 0.2:
                color = "#ff00ff" # Magenta for surprise

            # Weight = strength sigma
            width = 1 + (5 * r.sigma)

            # Label includes type and metrics
            label = f"{r.tau.name}\nσ={r.sigma:.2f}, κ={r.kappa:.2f}, ε={r.epsilon:.2f}"

            net.add_edge(
                r.src_id,
                r.tgt_id,
                value=width,
                color=color,
                title=label,
                label=r.tau.name if r.epsilon > 0.5 else "" # Only show label on high surprise
            )

        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100, "springStrength": 0.08 },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": { "iterations": 150 }
          },
          "edges": { "color": { "inherit": false }, "smooth": { "type": "continuous" } },
          "nodes": { "font": { "strokeWidth": 2 } }
        }
        """)

        net.save_graph(filename)
        print(f"Visualized field exported to: {filename}")
