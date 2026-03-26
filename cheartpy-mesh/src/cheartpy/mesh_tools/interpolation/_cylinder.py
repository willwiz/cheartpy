from typing import TYPE_CHECKING

import numpy as np

from cheartpy.mesh import CheartMesh, CheartMeshSpace

from ._remeshing import create_quad_boundary, create_quad_top_and_map

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A1, A2


def mean_cylindrical[T: np.floating](mat: A2[T]) -> A1[T]:
    """Return the cylindrical coordinates mapped from the cartesian coordinates.

    Difference in Angle between two vectors is more robustly given by:
    arctan( |a x b|, a . b )
    This is also equation have the angular dimension of the element
    For C1 continuity, the center node for elements are not place at the radius but rather
    r(q/2) = r(0|q) * (3 + cos(2 * dq))/ (4 * cos(dq))
    this converges to r with element refinement.
    """
    c = np.cos(mat[:, 1])
    s = np.sin(mat[:, 1])
    cq = c.mean()
    sq = s.mean()
    q = np.arctan2(sq, cq)
    sin = c * sq - s * cq
    cos = c * cq + s * sq
    dq = np.abs(np.arctan2(sin, cos)).mean()
    r = mat[:, 0].mean() * (3.0 + np.cos(2.0 * dq)) / (4.0 * np.cos(dq))
    return np.array([r, q, mat[:, 2].mean()])


def create_quad_space_cylindrical[T: np.floating](
    quad_map: Mapping[frozenset[int], int],
    x: CheartMeshSpace[T],
) -> CheartMeshSpace[T]:
    new_space = np.zeros((len(quad_map), x.v.shape[1]), dtype=float)
    for elem, i in quad_map.items():
        nodes = x.v[list(elem)]
        new_space[i] = mean_cylindrical(nodes)
    return CheartMeshSpace(len(new_space), new_space)


def create_quad_mesh_from_lin_cylindrical[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> CheartMesh[F, I]:
    top, quad_map = create_quad_top_and_map(mesh.top, mesh.space.n)
    boundary = create_quad_boundary(quad_map, mesh.bnd) if mesh.bnd else None
    space = create_quad_space_cylindrical(quad_map, mesh.space)
    return CheartMesh(space, top, boundary)
