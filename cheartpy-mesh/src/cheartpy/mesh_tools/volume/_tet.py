from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces._types import CheartEnum
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1

    from cheartpy.mesh import CheartMesh


def calculate_volume_tetrahedron_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> Result[float]:
    """Calculate the volume of a tetrahedron mesh.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The tetrahedron mesh.

    Returns
    -------
    float
        The volume of the tetrahedron mesh.

    """
    if mesh.top.TYPE is not CheartEnum.TETRAHEDRON1:
        return Err(ValueError("Mesh is not a tetrahedron mesh"))
    v = mesh.space.v[mesh.top.v]
    v0 = v[:, 0, :]
    v1 = v[:, 1, :]
    v2 = v[:, 2, :]
    v3 = v[:, 3, :]
    a: A1[F] = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0)
    vol = float(np.sum(np.abs(a)) / 6.0)
    return Ok(vol)


def calculate_tetrahedron_volume_by_surface[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], surfaces: Sequence[int]
) -> Result[float]:
    """Calculate the volume of a tetrahedron mesh by its surface.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The tetrahedron mesh.
    surfaces : Sequence[int]
        The surface IDs to use for the volume calculation.

    Returns
    -------
    float
        The volume of the tetrahedron mesh.

    """
    if mesh.top.TYPE is not CheartEnum.TETRAHEDRON1:
        return Err(ValueError("Mesh is not a tetrahedron mesh"))
    if mesh.bnd is None:
        return Err(ValueError("Mesh has no boundary"))
    bnd = mesh.bnd.v
    if not all(s in bnd for s in surfaces):
        return Err(ValueError("Some surfaces are not in the boundary"))
    vol = 0.0
    for surf_id in surfaces:
        surf = bnd[surf_id]
        v = mesh.space.v[surf.v]
        v0 = v[:, 0, :]
        v1 = v[:, 1, :]
        v2 = v[:, 2, :]
        a: A1[F] = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v0)
        vol += float(np.sum(a) / 6.0)
    return Ok(vol)
