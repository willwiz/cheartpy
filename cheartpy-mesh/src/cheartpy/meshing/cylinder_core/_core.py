from typing import TYPE_CHECKING, overload

import numpy as np

from cheartpy.mesh import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from cheartpy.mesh_tools import IndexPermutation, recompile_cheart_mesh
from cheartpy.mesh_tools.surface_core import relabel_cheart_surfaces

from ._types import CartesianDirection

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A2, ToFloat


__all__ = [
    "_convert_reference_space_to_cylindrical",
    "cylindrical_to_cartesian",
    "gen_end_node_mapping",
    "merge_circ_ends",
    "reference_to_cylindrical",
    "rotate_axis",
    "update_boundary",
    "update_elems",
]


def create_end_wrap_permulation[I: np.integer](
    left: CheartMeshPatch[I],
    right: CheartMeshPatch[I],
    size: int,
) -> IndexPermutation[I]:
    """Create an index permutation that wraps the end nodes of a mesh patch to another patch.

    A linear linear permutation where only the end nodes of the right patch are replaced with the
    left patch's end nodes. The rest of the nodes are unchanged.

    Parameters
    ----------
    left : CheartMeshPatch[I]
        The left mesh patch whose end nodes will be used for wrapping.
    right : CheartMeshPatch[I]
        The right mesh patch whose end nodes will be replaced.
    size : int
        The total number of nodes in the mesh.

    Returns
    -------
    IndexPermutation[I]
        Resulting index permutation for updating nodes and elements.

    """
    fwd = np.arange(size, dtype=right.v.dtype)
    for i in range(left.n):
        for j, k in {0: 0, 1: 2, 2: 1, 3: 3}.items():
            fwd[right.v[i, j]] = left.v[i, k]
    old = np.unique(fwd, sorted=True)
    new = np.arange(len(old), dtype=old.dtype)
    return IndexPermutation(old=old, new=new, fwd=fwd)


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
    perm = create_end_wrap_permulation(cube.bnd.v[3], cube.bnd.v[4], cube.space.n)
    # node_map = gen_end_node_mapping(cube.bnd.v[3], cube.bnd.v[4])
    # new_t = update_elems(cube.top.v, node_map)
    # new_b = {
    #     n: update_boundary(cube.bnd.v[k], node_map, n) for n, k in {3: 1, 4: 2, 1: 5, 2: 6}.items()
    # }
    cube = relabel_cheart_surfaces(cube, {1: 3, 2: 4, 5: 1, 6: 2}, closed=True).unwrap()
    mesh = CheartMesh(
        cube.space,
        CheartMeshTopology(cube.top.n, perm.fwd[cube.top.v], cube.top.TYPE),
        CheartMeshBoundary(
            cube.bnd.n,
            {
                tag: CheartMeshPatch(tag, b.n, b.k, perm.fwd[b.v], b.TYPE)
                for tag, b in cube.bnd.v.items()
            },
            cube.bnd.TYPE,
        ),
    )
    return recompile_cheart_mesh(mesh)


def _convert_reference_space_to_cylindrical[F: np.floating](
    x: A2[F], r_in: ToFloat, r_out: ToFloat, length: ToFloat, base: ToFloat
) -> A2[F]:
    r = np.zeros_like(x)
    r[:, 0] = (r_out - r_in) * x[:, 0] ** 0.8165 + r_in
    r[:, 1] = 2.0 * np.pi * x[:, 1]
    r[:, 2] = length * x[:, 2] + base
    return r


def reference_to_cylindrical[F: np.floating, I: np.integer](
    cube: CheartMesh[F, I], r_in: ToFloat, r_out: ToFloat, length: ToFloat, base: ToFloat
) -> CheartMesh[F, I]:
    new_x = _convert_reference_space_to_cylindrical(cube.space.v, r_in, r_out, length, base)
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


def _get_rotation_matrix(orientation: CartesianDirection) -> A2[np.intp]:
    match orientation:
        case CartesianDirection.x:
            return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.intp) @ np.array(
                [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.intp
            )
        case CartesianDirection.y:
            return np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.intp) @ np.array(
                [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.intp
            )
        case CartesianDirection.z:
            return np.eye(3, dtype=np.intp)


def _rotate_axis_space[F: np.floating](x: A2[F], orientation: CartesianDirection) -> A2[F]:
    return np.matmul(x, _get_rotation_matrix(orientation).T)


def _rotate_axis_mesh[F: np.floating, I: np.integer](
    g: CheartMesh[F, I], orientation: CartesianDirection
) -> CheartMesh[F, I]:
    return CheartMesh(
        CheartMeshSpace(g.space.n, _rotate_axis_space(g.space.v, orientation)),
        g.top,
        g.bnd,
    )


@overload
def rotate_axis[F: np.floating, I: np.integer](
    g: CheartMesh[F, I], orientation: CartesianDirection
) -> CheartMesh[F, I]: ...
@overload
def rotate_axis[F: np.floating](g: A2[F], orientation: CartesianDirection) -> A2[F]: ...
def rotate_axis[F: np.floating, I: np.integer](
    g: CheartMesh[F, I] | A2[F], orientation: CartesianDirection
):
    match g:
        case CheartMesh():
            return _rotate_axis_mesh(g, orientation)
        case np.ndarray():
            return _rotate_axis_space(g, orientation)
