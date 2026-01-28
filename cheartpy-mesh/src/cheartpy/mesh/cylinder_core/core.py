from typing import TYPE_CHECKING

import numpy as np
from cheartpy.mesh.struct import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from cheartpy.mesh.validation import remove_dangling_nodes
from cheartpy.vtk.types import VtkEnum

from .data import CartesianDirection

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A2


__all__ = [
    "convert_cartesian_space_to_cylindrical",
    "convert_to_cylindrical",
    "cylindrical_to_cartesian",
    "gen_end_node_mapping",
    "merge_circ_ends",
    "rotate_axis",
    "update_boundary",
    "update_elems",
]


def gen_end_node_mapping[I: np.integer](
    left: CheartMeshPatch[I],
    right: CheartMeshPatch[I],
) -> Mapping[int, int]:
    node_map: dict[int, int] = {}
    for i in range(left.n):
        for j, k in {0: 0, 1: 2, 2: 1, 3: 3}.items():
            node_map[right.v[i, j]] = left.v[i, k]
    return node_map


def update_elems[I: np.integer](elems: A2[I], end_map: Mapping[int, int]) -> A2[I]:
    new_elems = elems.copy()
    for i, row in enumerate(elems):
        for j, v in enumerate(row):
            if v in end_map:
                new_elems[i, j] = end_map[int(v)]
    return new_elems


def update_boundary[I: np.integer](
    patch: CheartMeshPatch[I],
    end_map: Mapping[int, int],
    tag: int,
) -> CheartMeshPatch[I]:
    surf = patch.v.copy()
    for i, row in enumerate(surf):
        for j, v in enumerate(row):
            if v in end_map:
                surf[i, j] = end_map[int(v)]
    return CheartMeshPatch(tag, patch.n, patch.k, surf, patch.TYPE)


def merge_circ_ends[F: np.floating, I: np.integer](cube: CheartMesh[F, I]) -> CheartMesh[F, I]:
    if cube.bnd is None:
        msg = "Mesh must have a boundary to merge circular ends."
        raise ValueError(msg)
    node_map = gen_end_node_mapping(cube.bnd.v[3], cube.bnd.v[4])
    new_t = update_elems(cube.top.v, node_map)
    new_b = {
        n: update_boundary(cube.bnd.v[k], node_map, n) for n, k in {3: 1, 4: 2, 1: 5, 2: 6}.items()
    }
    mesh = CheartMesh(
        cube.space,
        CheartMeshTopology(len(new_t), new_t, VtkEnum.LinHexahedron),
        CheartMeshBoundary(len(new_b), new_b, VtkEnum.LinQuadrilateral),
    )
    return remove_dangling_nodes(mesh)


def convert_cartesian_space_to_cylindrical[F: np.floating](
    x: A2[F],
    r_in: float,
    r_out: float,
    length: float,
    base: float,
) -> A2[F]:
    r = np.zeros_like(x)
    r[:, 0] = (r_out - r_in) * x[:, 0] ** 0.707 + r_in
    r[:, 1] = 2.0 * np.pi * x[:, 1]
    r[:, 2] = length * x[:, 2] + base
    return r


def convert_to_cylindrical[F: np.floating, I: np.integer](
    cube: CheartMesh[F, I],
    r_in: float,
    r_out: float,
    length: float,
    base: float,
) -> CheartMesh[F, I]:
    new_x = convert_cartesian_space_to_cylindrical(
        cube.space.v,
        r_in,
        r_out,
        length,
        base,
    )
    return CheartMesh(CheartMeshSpace(len(new_x), new_x), cube.top, cube.bnd)


def cylindrical_to_cartesian[F: np.floating, I: np.integer](
    g: CheartMesh[F, I],
) -> CheartMesh[F, I]:
    cart_space = np.zeros_like(g.space.v)
    radius = g.space.v[:, 0]
    theta = g.space.v[:, 1]
    cart_space[:, 0] = radius * np.cos(theta)
    cart_space[:, 1] = radius * np.sin(theta)
    cart_space[:, 2] = g.space.v[:, 2]
    return CheartMesh(CheartMeshSpace(g.space.n, cart_space), g.top, g.bnd)


def rotate_axis[F: np.floating, I: np.integer](
    g: CheartMesh[F, I],
    orientation: CartesianDirection,
) -> CheartMesh[F, I]:
    match orientation:
        case CartesianDirection.x:
            mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ np.array(
                [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            )
        case CartesianDirection.y:
            mat = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]) @ np.array(
                [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            )
        case CartesianDirection.z:
            return g
    return CheartMesh(CheartMeshSpace(g.space.n, g.space.v @ mat.T), g.top, g.bnd)
