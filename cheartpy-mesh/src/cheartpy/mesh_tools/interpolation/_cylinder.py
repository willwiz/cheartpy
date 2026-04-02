from typing import TYPE_CHECKING

import numpy as np

from cheartpy.mesh import CheartMesh, CheartMeshSpace

from ._remeshing import create_quad_boundary, create_quad_top_and_map

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pytools.arrays import A1, A2


def mean_cylindrical_position[F: np.floating](
    radius: A2[F], cosine: A1[F], sine: A1[F], z: A2[F]
) -> Sequence[F]:
    """Return the cylindrical coordinates mapped from the cartesian coordinates.

    Difference in Angle between two vectors is more robustly given by:
    arctan( |a x b|, a . b )
    This is also equation have the angular dimension of the element
    For C1 continuity, the center node for elements are not place at the radius but rather
    r(q/2) = r(0|q) * (3 + cos(2 * dq))/ (4 * cos(dq))
    this converges to r with element refinement.
    """
    cq = cosine.mean()
    sq = sine.mean()
    q = np.arctan2(sq, cq)
    sin = cosine * sq - sine * cq
    cos = cosine * cq + sine * cq
    dq = np.abs(np.arctan2(sin, cos)).mean()
    r = radius.mean() * (3.0 + np.cos(2.0 * dq)) / (4.0 * np.cos(dq))
    return r, q, z.mean()


def create_quad_space_cylinder[F: np.floating](
    quad_map: Mapping[frozenset[int], int],
    x: CheartMeshSpace[F],
) -> CheartMeshSpace[F]:
    cosine = np.cos(x.v[:, 1])
    sine = np.sin(x.v[:, 1])
    elem_nodes = (np.array(list(elem)) for elem in quad_map)
    new_space = np.fromiter(
        (mean_cylindrical_position(x.v[n, 0], cosine[n], sine[n], x.v[n, 2]) for n in elem_nodes),
        dtype=f"3{x.v.dtype.str}",
    )
    return CheartMeshSpace(len(new_space), new_space)


def create_quad_mesh_from_lin_cylindrical[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> CheartMesh[F, I]:
    top, quad_map = create_quad_top_and_map(mesh.top, mesh.space.n)
    boundary = create_quad_boundary(quad_map, mesh.bnd) if mesh.bnd else None
    space = create_quad_space_cylinder(quad_map, mesh.space)
    return CheartMesh(space, top, boundary)
