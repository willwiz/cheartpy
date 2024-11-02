import os
import numpy as np
from scipy.linalg import lstsq
from ...var_types import *
from ...tools.basiclogging import _Logger, NullLogger
from .data import *
from .elements import *


def compute_normal_patch(
    basis: Mat[f64],
    space: Mat[f64],
    elem: Vec[i32],
    ref_space: Mat[f64],
    LOG: _Logger = NullLogger(),
):
    nodes = space[elem] - ref_space
    F = np.array(
        [[nodes[:, i] @ basis[j] for j in range(3)] for i in range(3)]
    ) + np.identity(3)
    # print(f"{F=}")
    res, *_ = lstsq(F.T, np.array([0, 0, 1], dtype=float), lapack_driver="gelsy")
    # print(f"{res=}")
    return res


def normalize_by_row(vals: Mat[f64]) -> Mat[f64]:
    norm = np.sqrt(np.einsum("...i,...i", vals, vals))
    return vals / norm[:, np.newaxis]


def compute_normal_surface_at_center(
    kind: VtkType, space: Mat[f64], elem: Mat[i32], LOG: _Logger = NullLogger()
):
    centroid = np.mean(kind.ref_nodes, axis=0)
    interp_basis = kind.shape_dfuncs(centroid)
    LOG.debug(f"{interp_basis=}")
    normals = np.array(
        [compute_normal_patch(interp_basis, space, i, kind.ref_nodes) for i in elem],
        dtype=float,
    )
    LOG.debug(f"{normals=}")
    return normalize_by_row(normals)


def compute_normal_surface_at_nodes(
    kind: VtkType, space: Mat[f64], elem: Mat[i32], LOG: _Logger = NullLogger()
):
    interp_basis = {k: kind.shape_dfuncs(v) for k, v in enumerate(kind.ref_nodes)}
    LOG.debug(f"{interp_basis=}")
    normals = {
        k: np.array(
            [compute_normal_patch(v, space, i, kind.ref_nodes) for i in elem],
            dtype=float,
        )
        for k, v in interp_basis.items()
    }
    return {k: normalize_by_row(v) for k, v in normals.items()}


def reset_cheart_mesh(mesh: CheartMesh): ...
