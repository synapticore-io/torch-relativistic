"""
Terrell-Penrose effect visualization — the TU Wien experiment, in code.

Reproduces the key finding of Schattschneider et al. (Communications
Physics, May 2025): a cube moving at relativistic speeds appears
*rotated*, not flattened. This script renders a 3D cube at several
fractions of c using the exact same physics that torch-relativistic's
transforms are built on, and validates that the library's
`terrell_rotation_angle` function returns the correct apparent rotation.

Usage:
    uv run python examples/terrell_penrose_demo.py

Output:
    examples/terrell_penrose_demo.html   (interactive 3D)
    examples/terrell_penrose_demo.png    (static screenshot for README)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch
from torch_relativistic.utils import terrell_rotation_angle


# ---------------------------------------------------------------------------
# Cube geometry
# ---------------------------------------------------------------------------

def unit_cube_vertices() -> np.ndarray:
    """8 vertices of a unit cube centered at origin, shape (8, 3)."""
    return np.array([
        [-0.5, -0.5, -0.5],
        [+0.5, -0.5, -0.5],
        [+0.5, +0.5, -0.5],
        [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5],
        [+0.5, -0.5, +0.5],
        [+0.5, +0.5, +0.5],
        [-0.5, +0.5, +0.5],
    ])


# 12 edges as pairs of vertex indices
CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # back face
    (4, 5), (5, 6), (6, 7), (7, 4),  # front face
    (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
]

# 6 faces as quads (vertex index lists), with colors
CUBE_FACES = [
    ([0, 1, 2, 3], "rgba(31, 119, 180, 0.6)"),   # back  (blue)
    ([4, 5, 6, 7], "rgba(255, 127, 14, 0.6)"),    # front (orange)
    ([0, 1, 5, 4], "rgba(44, 160, 44, 0.6)"),     # bottom (green)
    ([2, 3, 7, 6], "rgba(214, 39, 40, 0.6)"),     # top (red)
    ([0, 3, 7, 4], "rgba(148, 103, 189, 0.6)"),   # left (purple)
    ([1, 2, 6, 5], "rgba(140, 86, 75, 0.6)"),     # right (brown)
]


# ---------------------------------------------------------------------------
# Terrell-Penrose transformation
# ---------------------------------------------------------------------------

def terrell_penrose_transform(vertices: np.ndarray, beta: float) -> np.ndarray:
    """
    Apply the Terrell-Penrose apparent transformation to vertex positions.

    An object moving along the x-axis with velocity v = beta * c, observed
    from far away along the +z axis, appears rotated by arcsin(beta) rather
    than Lorentz-contracted. This is because light from the back of the object
    (smaller z, further from observer) was emitted earlier, when the object
    was at a different x position.

    For the far-field approximation (observer at +z infinity):

        apparent_x = x / gamma + beta * (z_max - z)
        apparent_y = y
        apparent_z = z

    where gamma = 1 / sqrt(1 - beta^2) and z_max is the front face z coord.

    This is mathematically equivalent to a rotation about the y-axis by
    angle theta = arcsin(beta).

    Args:
        vertices: (N, 3) array of (x, y, z) positions
        beta: velocity as fraction of c, in [0, 1)

    Returns:
        (N, 3) array of apparent positions
    """
    if beta < 1e-12:
        return vertices.copy()

    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    z_max = vertices[:, 2].max()

    result = vertices.copy()
    # Lorentz contraction along x
    result[:, 0] = vertices[:, 0] / gamma
    # Light travel time correction: back points appear shifted in x
    result[:, 0] += beta * (z_max - vertices[:, 2])
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def add_cube_to_figure(
    fig: go.Figure,
    vertices: np.ndarray,
    row: int,
    col: int,
    title: str,
) -> None:
    """Add a wireframe + transparent-face cube to a subplot."""
    # Edges
    for i, j in CUBE_EDGES:
        fig.add_trace(
            go.Scatter3d(
                x=[vertices[i, 0], vertices[j, 0]],
                y=[vertices[i, 1], vertices[j, 1]],
                z=[vertices[i, 2], vertices[j, 2]],
                mode="lines",
                line=dict(color="black", width=3),
                showlegend=False,
            ),
            row=row, col=col,
        )

    # Faces
    for face_idx, color in CUBE_FACES:
        v = vertices[face_idx]
        # Close the quad
        v_closed = np.vstack([v, v[:1]])
        fig.add_trace(
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=[0, 0], j=[1, 2], k=[2, 3],
                color=color,
                flatshading=True,
                showlegend=False,
            ),
            row=row, col=col,
        )

    # Vertices as dots
    fig.add_trace(
        go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            marker=dict(size=4, color="black"),
            showlegend=False,
        ),
        row=row, col=col,
    )


def main() -> None:
    velocities = [0.0, 0.5, 0.8, 0.95]
    n = len(velocities)

    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "scene"}] * n],
        subplot_titles=[
            f"v = {v:.2f}c  (rotation = {math.degrees(math.asin(v)):.1f}°)"
            if v > 0 else "v = 0 (at rest)"
            for v in velocities
        ],
        horizontal_spacing=0.02,
    )

    base_verts = unit_cube_vertices()

    for i, beta in enumerate(velocities):
        apparent = terrell_penrose_transform(base_verts, beta)
        label = f"v={beta:.2f}c" if beta > 0 else "rest"
        add_cube_to_figure(fig, apparent, row=1, col=i + 1, title=label)

    # Validate against torch-relativistic
    print("Validation: torch-relativistic vs analytical arcsin(v/c)")
    print(f"{'v/c':>6s} | {'arcsin (deg)':>12s} | {'library (deg)':>13s} | {'match':>5s}")
    print("-" * 50)
    for v in velocities:
        if v < 1e-12:
            continue
        analytical = math.degrees(math.asin(v))
        library_val = math.degrees(terrell_rotation_angle(torch.tensor(v)).item())
        match = abs(analytical - library_val) < 0.01
        print(f"{v:6.2f} | {analytical:12.4f} | {library_val:13.4f} | {'yes' if match else 'NO':>5s}")

    # Layout
    camera = dict(eye=dict(x=1.5, y=1.5, z=1.0))
    axis_range = [-1.2, 1.2]
    for i in range(1, n + 1):
        scene_name = f"scene{i}" if i > 1 else "scene"
        fig.layout[scene_name].update(
            xaxis=dict(range=axis_range, title="x (motion)"),
            yaxis=dict(range=axis_range, title="y"),
            zaxis=dict(range=axis_range, title="z (toward observer)"),
            camera=camera,
            aspectmode="cube",
        )

    fig.update_layout(
        width=1600,
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    out_dir = Path("examples")
    out_dir.mkdir(exist_ok=True)

    html_path = out_dir / "terrell_penrose_demo.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"\nWrote {html_path}")

    try:
        png_path = out_dir / "terrell_penrose_demo.png"
        fig.write_image(str(png_path), scale=2)
        print(f"Wrote {png_path}")
    except Exception as e:
        print(f"PNG export skipped ({e}) — install kaleido for static images")

    print("\nDone. Open the HTML file in a browser for interactive 3D.")


if __name__ == "__main__":
    main()
