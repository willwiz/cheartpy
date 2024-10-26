from .data import *
from .elements import *
from ...var_types import *
import numpy as np
from scipy.linalg import lstsq


def compute_normal_patch(
    basis: Mat[f64], space: Mat[f64], elem: Vec[i32], ref_space: Mat[f64]
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


def compute_normal_surface(kind: VtkElemInterface, space: Mat[f64], elem: Mat[i32]):
    ref_nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    centroid = np.mean(ref_nodes, axis=0)
    interp_basis = kind.shape_dfuncs(centroid)
    # print(f"{interp_basis=}")
    normals = np.array(
        [compute_normal_patch(interp_basis, space, i, ref_nodes) for i in elem],
        dtype=float,
    )
    # print(f"{normals=}")
    return normalize_by_row(normals)


def reset_cheart_mesh(mesh: CheartMesh): ...
