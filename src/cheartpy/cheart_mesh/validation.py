__all__ = ["reset_cheart_mesh"]
import numpy as np
from typing import Mapping
from ..var_types import *
from .data import *


def create_node_map(elems: Mat[i32]) -> Mapping[int, int]:
    node_map: dict[int, int] = dict()
    nn = 0
    for elem in elems:
        for node in elem:
            node_map[node] = nn
    return node_map


def create_elem_map(elems: Vec[i32]) -> Mapping[int, int]:
    return {k: v for v, k in enumerate(elems)}


def _create_new_bnd(
    p: _CheartMeshPatch,
    node_map: Mapping[int, int],
    elem_map: Mapping[int, int],
    elem: Vec[i32],
):
    keys = np.isin(p.k, elem)
    new_v = np.array([[node_map[i] for i in patch] for patch in p.v[keys]], dtype=int)
    new_k = np.array([elem_map[k] for k in p.k[keys]], dtype=int)
    return _CheartMeshPatch(p.tag, len(new_k), new_k, new_v)


def reset_cheart_mesh(g: CheartMesh, elems: Vec[i32]):
    new_elems = g.top.v[elems]
    node_map = create_node_map(new_elems)
    elem_map = create_elem_map(new_elems)
    new_x = g.space.v[list(node_map.keys())]
    new_t = np.ascontiguousarray(
        [[node_map[i] for i in elem] for elem in new_elems], dtype=int
    )

    boundary = (
        _CheartMeshBoundary(
            g.bnd.n,
            {
                k: _create_new_bnd(v, node_map, elem_map, elems)
                for k, v in g.bnd.v.items()
            },
            g.bnd.TYPE,
        )
        if g.bnd
        else None
    )
    return CheartMesh(
        _CheartMeshSpace(len(new_x), new_x),
        _CheartMeshTopology(len(new_t), new_t, g.top.TYPE),
        boundary,
    )
