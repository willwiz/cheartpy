import numpy as np
from typing import Mapping
from ...var_types import *
from ...cheart_mesh.data import *
from .maps import L2QMAP, L2QMAPDICT, L2QTYPEDICT


def gen_quadtop_node_sets(map: L2QMAP, top: Vec[i32]) -> Mapping[int, frozenset[int]]:
    return {k: frozenset([top[i] for i in elem]) for k, elem in enumerate(map)}


def create_quad_top_and_map(
    t: CheartMeshTopology,
) -> tuple[CheartMeshTopology, Mapping[frozenset[int], int]]:
    L2Q = L2QMAPDICT.get(t.TYPE)
    if L2Q is None:
        raise ValueError(f"topology to be interpolated must be linear")
    QUAD_TYPE = L2QTYPEDICT.get(t.TYPE)
    if QUAD_TYPE is None:
        raise ValueError(f"topology to be interpolated must be linear")
    quad_map: Mapping[frozenset[int], int] = dict()
    new_top = np.zeros((t.n, len(QUAD_TYPE.connectivity)), dtype=int)
    nn = 0
    for i, elem in enumerate(t.v):
        for j, v in gen_quadtop_node_sets(L2Q, elem).items():
            if v not in quad_map:
                quad_map[v] = nn
                nn = nn + 1
            new_top[i, j] = quad_map[v]
    return CheartMeshTopology(len(new_top), new_top, QUAD_TYPE), quad_map


def create_quad_surf(
    l2q: L2QMAP, quad_map: Mapping[frozenset[int], int], b: CheartMeshPatch
):
    new_bnd = np.zeros((b.n, len(l2q)), dtype=int)
    for i, elem in enumerate(b.v):
        for j, v in gen_quadtop_node_sets(l2q, elem).items():
            if v not in quad_map:
                raise ValueError("boundary node not found in map from topology")
            new_bnd[i, j] = quad_map[v]
    return CheartMeshPatch(b.tag, b.n, b.k, new_bnd)


def create_quad_boundary(quad_map: Mapping[frozenset[int], int], b: CheartMeshBoundary):
    L2Q = L2QMAPDICT.get(b.TYPE)
    if L2Q is None:
        raise ValueError(f"boundary to be interpolated must be linear")
    QUAD_TYPE = L2QTYPEDICT.get(b.TYPE)
    if QUAD_TYPE is None:
        raise ValueError(f"topology to be interpolated must be linear")
    surfs = {k: create_quad_surf(L2Q, quad_map, v) for k, v in b.v.items()}
    return CheartMeshBoundary(b.n, surfs, QUAD_TYPE)


def create_quad_space(quad_map: Mapping[frozenset[int], int], x: CheartMeshSpace):
    new_space = np.zeros((len(quad_map), x.v.shape[1]), dtype=float)
    for elem, i in quad_map.items():
        nodes = x.v[[k for k in elem]]
        new_space[i] = nodes.mean(axis=0)
    return CheartMeshSpace(len(new_space), new_space)


def create_quad_mesh_from_lin(mesh: CheartMesh):
    top, quad_map = create_quad_top_and_map(mesh.top)
    boundary = create_quad_boundary(quad_map, mesh.bnd) if mesh.bnd else None
    space = create_quad_space(quad_map, mesh.space)
    return CheartMesh(space, top, boundary)
