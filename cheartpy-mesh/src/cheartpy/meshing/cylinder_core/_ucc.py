"""Universal Cylindrical Space (UCS) functions for cylindrical CheartMesh."""

import dataclasses as dc
from typing import TYPE_CHECKING, Literal, Required, TypedDict

import numpy as np

from cheartpy.mesh_tools import normalize_by_row
from cheartpy.mesh_tools.interpolation import create_quad_mesh_from_lin
from cheartpy.meshing.hex_core import create_hex_mesh

from ._core import merge_circ_ends

if TYPE_CHECKING:
    from pytools.arrays import A1, A2

    from cheartpy.mesh import CheartMesh


class ShapeParameters(TypedDict, total=True):
    n_r: Required[int]
    n_c: Required[int]
    n_z: Required[int]
    axis: Literal["x", "y", "z"]


@dc.dataclass(slots=True, frozen=True)
class UCC[F: np.floating = np.floating]:
    """Universal Cylindrical Coordinates (UCC) for a cylindrical mesh.

    Attributes
    ----------
    a_r : A1[F]
        Radial coordinates.
    a_c : A1[F]
        Circumferential coordinates.
    a_z : A1[F]
        Axial coordinates.
    v_r : A2[F]
        Radial vector components.
    v_c : A2[F]
        Circumferential vector components.
    v_z : A2[F]
        Axial vector components.

    """

    a_r: A1[F]
    a_c: A1[F]
    a_z: A1[F]
    v_r: A2[F]
    v_c: A2[F]
    v_z: A2[F]


def generate_ucc_for_cylindrical_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    shape: ShapeParameters,
    *,
    quad: bool = False,
) -> UCC[F]:
    """Generate Universal Cylindrical Coordinates (UCC) for a cylindrical mesh.

    Parameters
    ----------
    mesh: CheartMesh[F, np.integer]
        The input cylindrical mesh for which to generate UCC.
    shape: ShapeParameters
        A dictionary containing the shape parameters of the cylindrical mesh:
        - n_r: Number of radial divisions.
        - n_c: Number of circumferential divisions.
        - n_z: Number of axial divisions.

    quad: bool, default = False
        If True, create a quadrilateral mesh from the cylindrical mesh.

    Returns
    -------
    UniversalCylindricalCoordinate[F]: A dataclass containing the following attributes:
        - a_r (A1[F]): Radial coordinates.
        - a_c (A1[F]): Circumferential coordinates.
        - a_z (A1[F]): Axial coordinates.
        - v_r (A1[F]): Radial vector components.
        - v_c (A1[F]): Circumferential vector components.
        - v_z (A1[F]): Axial vector components.

    """
    base_mesh = create_hex_mesh(dim=(shape["n_r"], shape["n_c"], shape["n_z"]))
    base_mesh = merge_circ_ends(base_mesh)
    if quad:
        base_mesh = create_quad_mesh_from_lin(base_mesh)
    axis = {"x": 0, "y": 1, "z": 2}[shape["axis"]]
    a_r = base_mesh.space.v[:, 0]
    a_c = base_mesh.space.v[:, 1]
    a_z = base_mesh.space.v[:, 2]
    v_z = np.zeros_like(mesh.space.v)
    v_z[:, axis] = 1.0
    v_r = mesh.space.v.copy()
    v_r[:, axis] = 0.0
    v_r = normalize_by_row(v_r).unwrap()
    v_c = np.cross(v_z, v_r)
    dtype = mesh.space.v.dtype
    return UCC(
        a_r=np.asarray(a_r, dtype=dtype),
        a_c=np.asarray(a_c, dtype=dtype),
        a_z=np.asarray(a_z, dtype=dtype),
        v_r=np.asarray(v_r, dtype=dtype),
        v_c=np.asarray(v_c, dtype=dtype),
        v_z=np.asarray(v_z, dtype=dtype),
    )
