__all__ = [
    "gen_end_node_mapping",
    "update_elems",
    "update_boundary",
    "merge_circ_ends",
    "convert_cartesian_space_to_cylindrical",
    "convert_to_cylindrical",
    "cylindrical_to_cartesian",
    "rotate_axis",
]
import numpy as np
from typing import Mapping
from ...var_types import *
from ...cheart_mesh import *
from .data import *


def gen_end_node_mapping(
    left: CheartMeshPatch, right: CheartMeshPatch
) -> Mapping[int, int]:
    """
    Assumes:
        x: r
        y: q
        z: z
    """
    node_map: dict[int, int] = dict()
    for i in range(left.n):
        for j, k in {0: 0, 1: 2, 2: 1, 3: 3}.items():
            node_map[right.v[i, j]] = left.v[i, k]
    return node_map


def update_elems(elems: Mat[int_t], map: Mapping[int, int]):
    new_elems = elems.copy()
    for i, row in enumerate(elems):
        for j, v in enumerate(row):
            if v in map:
                new_elems[i, j] = map[v]
    return new_elems


def update_boundary(patch: CheartMeshPatch, map: Mapping[int, int], tag: int):
    surf = patch.v.copy()
    for i, row in enumerate(surf):
        for j, v in enumerate(row):
            if v in map:
                surf[i, j] = map[v]
    return CheartMeshPatch(tag, patch.n, patch.k, surf)


def merge_circ_ends(cube: CheartMesh):
    if cube.bnd is None:
        raise
    node_map = gen_end_node_mapping(cube.bnd.v[3], cube.bnd.v[4])
    new_t = update_elems(cube.top.v, node_map)
    new_b: dict[int | str, CheartMeshPatch] = {
        n: update_boundary(cube.bnd.v[k], node_map, n)
        for n, k in {3: 1, 4: 2, 1: 5, 2: 6}.items()
    }
    mesh = CheartMesh(
        cube.space,
        CheartMeshTopology(len(new_t), new_t, VtkType.HexahedronLinear),
        CheartMeshBoundary(len(new_b), new_b, VtkType.QuadrilateralLinear),
    )
    return remove_dangling_nodes(mesh)


def convert_cartesian_space_to_cylindrical(
    x: Mat[f64], r_in: float, r_out: float, length: float, base: float
):
    r = np.zeros_like(x)
    r[:, 0] = (r_out - r_in) * x[:, 0] ** 0.707 + r_in
    r[:, 1] = 2.0 * np.pi * x[:, 1]
    r[:, 2] = length * x[:, 2] + base
    return r


def convert_to_cylindrical(
    cube: CheartMesh, r_in: float, r_out: float, length: float, base: float
):
    new_x = convert_cartesian_space_to_cylindrical(
        cube.space.v, r_in, r_out, length, base
    )
    return CheartMesh(CheartMeshSpace(len(new_x), new_x), cube.top, cube.bnd)


def cylindrical_to_cartesian(g: CheartMesh) -> CheartMesh:
    cart_space = np.zeros_like(g.space.v)
    radius = g.space.v[:, 0]
    theta = g.space.v[:, 1]
    cart_space[:, 0] = radius * np.cos(theta)
    cart_space[:, 1] = radius * np.sin(theta)
    cart_space[:, 2] = g.space.v[:, 2]
    return CheartMesh(CheartMeshSpace(g.space.n, cart_space), g.top, g.bnd)


def rotate_axis(g: CheartMesh, orientation: CartesianDirection) -> CheartMesh:
    match orientation:
        case CartesianDirection.x:
            mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ np.array(
                [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
            )
        case CartesianDirection.y:
            mat = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]) @ np.array(
                [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
            )
        case CartesianDirection.z:
            return g
    return CheartMesh(CheartMeshSpace(g.space.n, g.space.v @ mat.T), g.top, g.bnd)
