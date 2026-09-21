from typing import TYPE_CHECKING
from warnings import deprecated

import numpy as np
from cheartpy.elem_interfaces import CheartEnum

from ._struct import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A2, ToInt


__all__ = ["remove_dangling_nodes"]


def create_node_map[T: np.integer](elems: A2[T]) -> Mapping[T, ToInt]:
    node_map: dict[T, int] = {}
    nn = 0
    for node in np.unique(elems):
        node_map[node] = nn
        nn = nn + 1
    return node_map


def _create_new_bnd[T: np.integer, B: CheartEnum](
    p: CheartMeshPatch[T, B],
    node_map: Mapping[T, ToInt],
) -> CheartMeshPatch[T, B]:
    new_v = np.array([[node_map[i] for i in patch] for patch in p.v], dtype=int)
    return CheartMeshPatch(p.tag, p.k, new_v, p.type)


@deprecated("Use cheartpy.mesh_tools.recompile_cheartmesh instead.")
def remove_dangling_nodes[F: np.floating, I: np.integer, T: CheartEnum, B: CheartEnum](
    g: CheartMesh[F, I, T, B],
) -> CheartMesh[F, I, T, B]:
    node_map = create_node_map(g.top.v)
    new_x = g.space.v[list(node_map.keys())]
    new_t = np.ascontiguousarray([[node_map[i] for i in elem] for elem in g.top.v], dtype=int)
    boundary = CheartMeshBoundary(
        {k: _create_new_bnd(v, node_map) for k, v in g.bnd.v.items()},
    )
    return CheartMesh(
        CheartMeshSpace(new_x),
        CheartMeshTopology(new_t, g.top.type),
        boundary,
    )
