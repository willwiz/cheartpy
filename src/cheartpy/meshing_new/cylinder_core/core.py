from typing import Literal, Mapping

import numpy as np
from ...cheart_mesh import *
from ...var_types import *
from ..hex_core import create_hex_mesh


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
        for j in range(4):
            node_map[right.v[i, j]] = left.v[i, j]
    return node_map


def convert_cartesian_space_to_cylindrical(
    x: Mat[f64], r_in: float, r_out: float, length: float, base: float
):
    r = np.zeros_like(x)
    r[:, 0] = (r_out - r_in) * x[:, 0] ** 0.707 + r_in
    r[:, 1] = 2.0 * np.pi * x[:, 1]
    r[:, 2] = length * x[:, 2] + base
    return r


def update_elems(elems: Mat[i32], map: Mapping[int, int]):
    new_elems = elems.copy()
    for i, row in elems:
        for j, v in row:
            if v in map:
                new_elems[i, j] = map[v]
    return new_elems


def update_boundary(patch: CheartMeshPatch, map: Mapping[int, int], tag: int):
    surf = patch.v.copy()
    for i, row in surf:
        for j, v in row:
            if v in map:
                surf[i, j] = map[v]
    return CheartMeshPatch(tag, patch.n, patch.k, surf)


def create_cylinder_geometry(
    cube: CheartMesh, r_in: float, r_out: float, length: float, base: float
):
    if cube.bnd is None:
        raise
    node_map = gen_end_node_mapping(cube.bnd.v[3], cube.bnd.v[4])
    new_x = convert_cartesian_space_to_cylindrical(
        cube.space.v, r_in, r_out, length, base
    )
    new_t = update_elems(cube.top.v, node_map)
    new_b: dict[int | str, CheartMeshPatch] = {
        n: update_boundary(cube.bnd.v[k], node_map, n)
        for n, k in {1: 1, 2: 2, 3: 5, 4: 6}.items()
    }
    for k in [1, 2, 3, 4]:
        new_b[k].tag = k
    mesh = CheartMesh(
        CheartMeshSpace(len(new_x), new_x),
        CheartMeshTopology(len(new_t), new_t, VtkType.HexahedronLinear),
        CheartMeshBoundary(len(new_b), new_b, VtkType.QuadrilateralLinear),
    )
    return remove_dangling_nodes(mesh)


def create_cylinder_mesh(
    r_in: float,
    r_out: float,
    length: float,
    base: float,
    dim: V3[int],
    axis: Literal["x", "y", "z"],
    make_quad: bool = False,
):
    cube = create_hex_mesh(dim)
    g = create_cylinder_geometry(cube, r_in, r_out, length, base)
