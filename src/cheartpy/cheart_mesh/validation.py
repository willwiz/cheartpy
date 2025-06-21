from __future__ import annotations

__all__ = ["remove_dangling_nodes"]

from typing import TYPE_CHECKING

import numpy as np

from .data import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from arraystubs import Arr2


def create_node_map[T: np.integer](elems: Arr2[T]) -> Mapping[T, int]:
    node_map: dict[T, int] = {}
    nn = 0
    for node in np.unique(elems):
        node_map[node] = nn
        nn = nn + 1
    return node_map


def _create_new_bnd[T: np.integer](
    p: CheartMeshPatch[T],
    node_map: Mapping[T, int],
) -> CheartMeshPatch[T]:
    new_v = np.array([[node_map[i] for i in patch] for patch in p.v], dtype=int)
    return CheartMeshPatch(p.tag, p.n, p.k, new_v)


def remove_dangling_nodes[F: np.floating, I: np.integer](g: CheartMesh[F, I]) -> CheartMesh[F, I]:
    node_map = create_node_map(g.top.v)
    new_x = g.space.v[list(node_map.keys())]
    new_t = np.ascontiguousarray([[node_map[i] for i in elem] for elem in g.top.v], dtype=int)
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
