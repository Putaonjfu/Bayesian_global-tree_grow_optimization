
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Set
import networkx as nx
import trimesh

from .io_utils import dedup_edges

def load_obj_edges(path: str) -> tuple[np.ndarray, List[Tuple[int,int]]]:
    """
    Load OBJ and return (vertices Nx3, edges list of (i,j) 0-indexed).
    Supports:
      - 'v' and 'l' (preferred)
      - faces 'f' converted to boundary edges
    """
    # Try fast path: manual parse for v & l
    verts: List[List[float]] = []
    edges: List[Tuple[int,int]] = []
    faces: List[List[int]] = []

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line:
                    continue
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("l "):
                    parts = line.strip().split()[1:]
                    if len(parts) >= 2:
                        # l may have more than 2, make path edges
                        idxs = [int(p.split("/")[0]) - 1 for p in parts]
                        for a, b in zip(idxs[:-1], idxs[1:]):
                            edges.append((a, b))
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    if len(parts) >= 3:
                        idxs = [int(p.split("/")[0]) - 1 for p in parts]
                        faces.append(idxs)
    except Exception:
        # fallback to trimesh if manual fails
        mesh = trimesh.load(path, force="mesh", skip_materials=True, process=False)
        v = mesh.vertices.view(np.ndarray)
        # convert faces to edges
        e = []
        for face in mesh.faces:
            ring = list(face) + [face[0]]
            e.extend([(ring[i], ring[i+1]) for i in range(len(face))])
        return v, dedup_edges(e)

    v = np.array(verts, dtype=float) if len(verts)>0 else None
    e = edges.copy()

    if v is None:
        # If no manual verts parsed, try trimesh
        mesh = trimesh.load(path, force="mesh", skip_materials=True, process=False)
        v = mesh.vertices.view(np.ndarray)

    # If no 'l' edges, convert faces to edges
    if len(e) == 0 and len(faces) > 0:
        for face in faces:
            ring = list(face) + [face[0]]
            e.extend([(ring[i], ring[i+1]) for i in range(len(face))])

    e = dedup_edges(e)
    if v is None or v.shape[0] == 0:
        raise ValueError(f"OBJ has no vertices: {path}")
    return v, e

def connected_components(vertices: np.ndarray, edges: List[Tuple[int,int]]):
    """Split into connected components. Return list of dicts: {vids:set, edges:list, centroid:np.ndarray}"""
    g = nx.Graph()
    # add vertices that appear in any edge
    vids_used: Set[int] = set()
    for i,j in edges:
        g.add_edge(i,j)
        vids_used.add(i); vids_used.add(j)

    comps = []
    if g.number_of_nodes()==0:
        return comps

    for cc in nx.connected_components(g):
        cc_vids: Set[int] = set(cc)
        cc_edges = [(i,j) for (i,j) in edges if i in cc_vids and j in cc_vids]
        if len(cc_edges)==0:
            continue
        pts = vertices[list(cc_vids)]
        centroid = pts.mean(axis=0) if pts.size>0 else np.zeros(3, dtype=float)
        comps.append({"vids": cc_vids, "edges": cc_edges, "centroid": centroid})
    return comps

def reindex_after_prune(vertices: np.ndarray, edges_kept: List[Tuple[int,int]]):
    """Keep only vertices referenced by edges_kept, reindex to dense [0..n-1]."""
    used: Set[int] = set()
    for a,b in edges_kept:
        used.add(a); used.add(b)
    if len(used)==0:
        # no edges; we drop all vertices to avoid orphan cloud in OBJ lines mode
        return np.zeros((0,3), dtype=float), []

    used_sorted = sorted(used)
    old2new: Dict[int,int] = {old:i for i,old in enumerate(used_sorted)}
    new_vertices = vertices[used_sorted].copy()
    new_edges = [(old2new[a], old2new[b]) for (a,b) in edges_kept]
    return new_vertices, new_edges
