__all__ = ["remove_dangling_nodes"]
import numpy as np
from typing import Mapping
from ..var_types import *
from .data import *


def create_node_map(elems: Mat[int_t]) -> Mapping[int, int]:
    node_map: dict[int, int] = dict()
    nn = 0
    for node in np.unique(elems):
        node_map[node] = nn
        nn = nn + 1
    return node_map


def _create_new_bnd(p: CheartMeshPatch, node_map: Mapping[int, int]):
    new_v = np.array([[node_map[i] for i in patch] for patch in p.v], dtype=int)
    return CheartMeshPatch(p.tag, p.n, p.k, new_v)


def remove_dangling_nodes(g: CheartMesh):
    """
    Assume Topology is complete
    """
    node_map = create_node_map(g.top.v)
    new_x = g.space.v[list(node_map.keys())]
    new_t = np.ascontiguousarray(
        [[node_map[i] for i in elem] for elem in g.top.v], dtype=int
    )
    boundary = (
        CheartMeshBoundary(
            g.bnd.n,
            {k: _create_new_bnd(v, node_map) for k, v in g.bnd.v.items()},
            g.bnd.TYPE,
        )
        if g.bnd
        else None
    )
    return CheartMesh(
        CheartMeshSpace(len(new_x), new_x),
        CheartMeshTopology(len(new_t), new_t, g.top.TYPE),
        boundary,
    )
