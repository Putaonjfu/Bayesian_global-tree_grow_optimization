
from __future__ import annotations
import numpy as np
import os
from typing import List, Tuple

def load_xyz(path: str, max_points: int | None = None) -> np.ndarray:
    """Load XYZ file (ascii). If columns > 3, take first 3. Optionally subsample to max_points."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"XYZ not found: {path}")
    pts = np.loadtxt(path, dtype=float, ndmin=2)
    if pts.shape[1] < 3:
        raise ValueError(f"XYZ must have >=3 columns, got shape {pts.shape} at {path}")
    pts = pts[:, :3]
    if max_points is not None and pts.shape[0] > max_points:
        # random subsample
        idx = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts

def dedup_edges(edges: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """Remove duplicates and self-loops, normalize so (i<j)."""
    s = set()
    out = []
    for i,j in edges:
        if i==j: 
            continue
        if i>j: i,j = j,i
        if (i,j) not in s:
            s.add((i,j))
            out.append((i,j))
    return out

def write_obj_lines(path: str, vertices: np.ndarray, edges: List[Tuple[int,int]]) -> None:
    """Write OBJ with only v + l (1-indexed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for (i,j) in edges:
            # OBJ is 1-indexed
            f.write(f"l {i+1} {j+1}\n")
