from typing import TYPE_CHECKING

import numpy as np

from cheartpy.mesh import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)

from ._maps import L2QMAP, L2QMAPDICT, L2QTYPEDICT

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A1, ToInt


def gen_quadtop_node_sets[T: np.integer](
    l2qmap: L2QMAP,
    top: A1[T],
) -> Mapping[int, frozenset[int]]:
    return {k: frozenset([top[i] for i in elem]) for k, elem in enumerate(l2qmap)}


def create_quad_top_and_map[I: np.integer](
    t: CheartMeshTopology[I],
    nn: ToInt,
) -> tuple[CheartMeshTopology[I], Mapping[frozenset[int], int]]:
    l2qmap = L2QMAPDICT.get(t.type)
    if l2qmap is None:
        msg = f"No map found for {t.type}. Topology to be interpolated must be linear"
        raise ValueError(msg)
    quad_elem = L2QTYPEDICT.get(t.type)
    if quad_elem is None:
        msg = f"No quad topology found for {t.type}.Topology to be interpolated must be linear"
        raise ValueError(msg)
    # quad_map: Mapping[frozenset[int], int] = dict()
    # nn = 0
    quad_map: Mapping[frozenset[int], int] = {frozenset([i]): i for i in range(nn)}
    new_top = np.zeros((t.n, len(quad_elem.connectivity)), dtype=int)
    for i, elem in enumerate(t.v):
        for j, v in gen_quadtop_node_sets(l2qmap, elem).items():
            if v not in quad_map:
                quad_map[v] = int(nn)
                nn = nn + 1
            new_top[i, j] = quad_map[v]
    return CheartMeshTopology(new_top, quad_elem.body), quad_map


def create_quad_surf[I: np.integer](
    l2q: L2QMAP,
    quad_map: Mapping[frozenset[int], int],
    b: CheartMeshPatch[I],
) -> CheartMeshPatch[I]:
    new_bnd = np.zeros((b.n, len(l2q)), dtype=int)
    for i, elem in enumerate(b.v):
        for j, v in gen_quadtop_node_sets(l2q, elem).items():
            if v not in quad_map:
                msg = f"Boundary node {v} not found in map from topology"
                raise ValueError(msg)
            new_bnd[i, j] = quad_map[v]
    return CheartMeshPatch(b.tag, b.k, new_bnd, b.type)


def create_quad_boundary[I: np.integer](
    quad_map: Mapping[frozenset[int], int],
    b: CheartMeshBoundary[I],
) -> CheartMeshBoundary[I]:
    if b.type is None:
        msg = "No boundary patch found."
        raise ValueError(msg)
    l2qmap = L2QMAPDICT.get(b.type)
    if l2qmap is None:
        msg = f"No L2Q map found for {b.type}. Boundary to be interpolated must be linear"
        raise ValueError(msg)
    quad_elem = L2QTYPEDICT.get(b.type)
    if quad_elem is None:
        msg = f"No quad type found for {b.type}. Boundary to be interpolated must be linear"
        raise ValueError(msg)
    surfs = {k: create_quad_surf(l2qmap, quad_map, v) for k, v in b.v.items()}
    return CheartMeshBoundary(surfs)


def create_quad_space_cartesian[T: np.floating](
    quad_map: Mapping[frozenset[int], int],
    x: CheartMeshSpace[T],
) -> CheartMeshSpace[T]:
    new_space = np.zeros((len(quad_map), x.v.shape[1]), dtype=float)
    for elem, i in quad_map.items():
        nodes = x.v[list(elem)]
        new_space[i] = nodes.mean(axis=0)
    return CheartMeshSpace(new_space)


def create_quad_mesh_from_lin[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> CheartMesh[F, I]:
    top, quad_map = create_quad_top_and_map(mesh.top, mesh.space.n)
    boundary = create_quad_boundary(quad_map, mesh.bnd)
    space = create_quad_space_cartesian(quad_map, mesh.space)
    return CheartMesh(space, top, boundary)
