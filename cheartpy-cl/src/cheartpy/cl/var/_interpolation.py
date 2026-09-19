import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.cl.mesh import create_centerline_partition

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cheartpy.cl.mesh import CLPartition
    from pytools.arrays import A1, A2


@dc.dataclass(slots=True, frozen=True)
class CLShapeFunctions[F: np.floating]:
    domain: Sequence[A1[np.bool_]]
    basis: Sequence[Sequence[A1[F]]]


def create_elem_basis_on_cl[F: np.floating](
    a_z: A1[F], part: CLPartition[F]
) -> CLShapeFunctions[F]: ...


def create_centerline_basis_func_by_elem[F: np.floating](
    a_z: A1[F], part: CLPartition[F]
) -> CLShapeFunctions[F]: ...


def interpolate_v_on_elem[F: np.floating](
    v: Sequence[A1[F]], basis: CLShapeFunctions[F]
) -> A2[F]: ...


def interp_v[F: np.floating, I: np.integer](
    a_z: A1[F], part: CLPartition[F], v: A2[F]
) -> A2[F]: ...


def interp_cl_var_to_volume[F: np.floating, I: np.integer](
    a_z: A1[F], part: int | A1[F] | CLPartition[F, I], *v: A2[F]
) -> list[A2[F]]:
    """Interpolate variables define CL to the volume.

    Parameters
    ----------
    a_z : A1[F]
        The z coordinates of the volume.
    part : CLDef[F] | CLPartition[F]
        The CL partition.
    *v : A2[F]
        The variables defined on the CL, with shape (n_cl, v.shape[1]

    Returns
    -------
    list[A2[F]]
        The interpolated variables on the volume, with shape (a_z.shape[0], v.shape[1]).

    """
    part = create_centerline_partition(part)
    raise NotImplementedError


def interp_cl_row_var_to_volume[F: np.floating, I: np.integer = np.intp](
    a_z: A1[F], part: int | A1[F] | CLPartition[F, I], *v: A2[F]
) -> list[A2[F]]:
    """Interpolate scalar row variables define CL to the volume.

    This function is similar to `interp_cl_var_to_volume`, but it assumes that the variables are
    defined as row vectors, i.e., each column of `v` corresponds to a value on a centerline node.
    For now, only 1 row variable is supported, i.e., v.shape[0] == 1.

    Parameters
    ----------
    a_z : A1[F]
        The z coordinates of the volume.
    part : CLDef[F] | CLPartition[F]
        The CL partition.
    *v : A2[F]
        The variables defined on the CL, with shape (1, n_cl).

    Returns
    -------
    list[A2[F]]
        The interpolated variables on the volume, with shape (a_z.shape[0], v.shape[1]).

    """
    part = create_centerline_partition(part)
    raise NotImplementedError
