from typing import TYPE_CHECKING

import numpy as np
from cheartpy.cl.mesh import CLDef, CLPartition, create_cl_partition

if TYPE_CHECKING:
    from pytools.arrays import A1, A2


def create_elem_basis_on_cl[F: np.floating](
    a_z: A1[F], left: A1[F], right: A1[F]
) -> tuple[A1[np.bool_], tuple[A1[F], A1[F]]]:
    domain = right - left
    domain_nodes = (a_z > left[1]) & (a_z < right[1])
    right_basis = (a_z[domain_nodes] - left[1]) / domain[1]
    left_basis = 1.0 - right_basis
    return domain_nodes, (left_basis, right_basis)


def interpolate_v_on_elem[F: np.floating](
    v: tuple[A1[F], A1[F]], basis: tuple[A1[F], A1[F]]
) -> A2[F]:
    return (v[0] * basis[0][:, None] + v[1] * basis[1][:, None]).astype(v[0].dtype)


def _interp_v[F: np.floating, I: np.integer](a_z: A1[F], part: CLPartition[F], v: A2[F]) -> A2[F]:
    """Interpolate the CL variables to the volume."""
    basis = {
        k: create_elem_basis_on_cl(a_z, left, right)
        for k, (left, right) in enumerate(zip(part.domain, part.domain[1:], strict=False))
    }
    res = np.zeros((len(a_z), v.shape[1]), dtype=v.dtype)
    for elem, (domain, b) in basis.items():
        res[domain] = interpolate_v_on_elem(v[elem], b)
    return res


def interp_cl_var_to_volume[F: np.floating, I: np.integer](
    a_z: A1[F], part: CLDef[F] | CLPartition[F], *v: A2[F]
) -> list[A2[F]]:
    """Interpolate variables define CL to the volume."""
    match part:
        case CLPartition(): ...  # fmt: skip
        case _:
            part = create_cl_partition(part)
    return [_interp_v(a_z, part, vi) for vi in v]
