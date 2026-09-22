"""Universal Cylindrical Space (UCS) functions for cylindrical CheartMesh."""

import enum
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from cheartpy.mesh import CheartMesh, CheartMeshSpace

if TYPE_CHECKING:
    from pytools.arrays import A1, A2


class CartesianDirection(enum.IntEnum):
    X = 0
    Y = 1
    Z = 2


def warp_space[F: np.floating](
    x: A2[F],
    long_axis: Literal["X", "Y", "Z"] = "X",
    bending_axis: Literal["X", "Y", "Z"] = "Y",
) -> A2[F]:
    """Warp a cylinder mesh along a specified axis.

    Given the long_axis and axis of bending in {0,1,2}, the remaining out of plane axis is
    {0,1,2} - {long_axis, axis}. Mathematically, this is equivalent to 3 - long_axis - axis.

    Parameters
    ----------
    x : A2[F]
        The input mesh coordinates.
    long_axis : Literal["X", "Y", "Z"]
        The longitudinal axis of the cylinder.
    bending_axis : Literal["X", "Y", "Z"]
        The axis along which the cylinder is bent.

    Returns
    -------
    A2[F]
        The warped mesh coordinates.

    """
    long = CartesianDirection[long_axis]
    bending = CartesianDirection[bending_axis]
    out = 3 - long - bending
    arc_radius = 2.0 * x[:, long].max() / np.pi
    theta = 0.5 * np.pi * (1.0 - x[:, long] / x[:, long].max())
    r = arc_radius + x[:, out]
    c = np.zeros_like(x)
    c[:, bending] = x[:, bending]
    c[:, long] = r * np.cos(theta)
    c[:, out] = r * np.sin(theta)
    return c


@overload
def warp_about_axis[F: np.floating](
    x: A2[F], long_axis: Literal["X", "Y", "Z"] = "X", bending_axis: Literal["X", "Y", "Z"] = "Y"
) -> A2[F]: ...
@overload
def warp_about_axis[F: np.floating, I: np.integer](
    x: CheartMeshSpace[F],
    long_axis: Literal["X", "Y", "Z"] = "X",
    bending_axis: Literal["X", "Y", "Z"] = "Y",
) -> CheartMeshSpace[F]: ...
@overload
def warp_about_axis[F: np.floating, I: np.integer](
    x: CheartMesh[F, I],
    long_axis: Literal["X", "Y", "Z"] = "X",
    bending_axis: Literal["X", "Y", "Z"] = "Y",
) -> CheartMesh[F, I]: ...
def warp_about_axis[F: np.floating, I: np.integer](
    x: A2[F] | CheartMeshSpace[F] | CheartMesh[F, I],
    long_axis: Literal["X", "Y", "Z"] = "X",
    bending_axis: Literal["X", "Y", "Z"] = "Y",
):
    """Warp a cylinder mesh along a specified axis.

    Given the long_axis and axis of bending in {0,1,2}, the remaining out of plane axis is
    {0,1,2} - {long_axis, axis}. Mathematically, this is equivalent to 3 - long_axis - axis.

    Parameters
    ----------
    x : A2[F]
        The input mesh coordinates.
    long_axis : Literal["X", "Y", "Z"]
        The longitudinal axis of the cylinder.
    bending_axis : Literal["X", "Y", "Z"]
        The axis along which the cylinder is bent.

    Returns
    -------
    A2[F]
        The warped mesh coordinates.

    """
    match x:
        case CheartMesh():
            return CheartMesh(
                space=warp_about_axis(x.space, long_axis=long_axis, bending_axis=bending_axis),
                top=x.top,
                bnd=x.bnd,
            )
        case CheartMeshSpace():
            return CheartMeshSpace(
                v=warp_about_axis(x.v, long_axis=long_axis, bending_axis=bending_axis)
            )
        case np.ndarray():
            return warp_space(x, long_axis=long_axis, bending_axis=bending_axis)


def warp_vector_about_axis[F: np.floating](
    x: A2[F],
    a_z: A1[F],
    long_axis: Literal["X", "Y", "Z"] = "X",
    bending_axis: Literal["X", "Y", "Z"] = "Y",
) -> A2[F]:
    """Warp a vector field along a specified axis.

    Parameters
    ----------
    x : A2[F]
        The input vector field.
    a_z : A1[F]
        The axial coordinate of the vector field.
    long_axis : Literal["X", "Y", "Z"]
        The longitudinal axis of the cylinder.
    bending_axis : Literal["X", "Y", "Z"]
        The axis along which the cylinder is bent.

    Returns
    -------
    A2[F]
        The warped vector field.

    """
    long = CartesianDirection[long_axis]
    bending = CartesianDirection[bending_axis]
    out = 3 - long - bending
    theta = 0.5 * np.pi * (1.0 - a_z)
    c = np.eye(3)[np.newaxis, :, :].repeat(x.shape[0], axis=0)
    c[:, long, long] = np.cos(theta)
    c[:, long, out] = -np.sin(theta)
    c[:, out, long] = np.sin(theta)
    c[:, out, out] = np.cos(theta)
    return np.einsum("ijk,ik->ij", c, x)
