import numpy as np
from collections import defaultdict
from typing import cast
from scipy.linalg import lstsq
from ...var_types import *
from ...tools.basiclogging import *
from ...cheart_mesh import *


def L2norm(x: Vec[f64]) -> float:
    return cast(float, x @ x)


def compute_normal_patch(
    basis: Mat[f64],
    space: Mat[f64],
    elem: Vec[i32],
    ref_space: Mat[f64],
    LOG: ILogger = NullLogger(),
) -> Vec[f64]:
    nodes = space[elem] - ref_space
    U = np.array([[nodes[:, i] @ basis[j] for j in range(3)] for i in range(3)])
    F = U + np.identity(3)
    if np.linalg.det(F) < 0.01:
        LOG.warn(f"Element node order is inverted.")
        F = U - np.identity(3)
    res, *_ = lstsq(F.T, np.array([0, 0, 1], dtype=float), lapack_driver="gelsy")
    return res


def normalize_by_row(vals: Mat[f64]) -> Mat[f64]:
    norm = np.sqrt(np.einsum("...i,...i", vals, vals))
    return vals / norm[:, np.newaxis]


def compute_normal_surface_at_center(
    kind: VtkType, space: Mat[f64], elem: Mat[i32], LOG: ILogger = NullLogger()
):
    centroid = np.mean(kind.ref_nodes, axis=0)
    interp_basis = kind.shape_dfuncs(centroid)
    normals = np.array(
        [compute_normal_patch(interp_basis, space, i, kind.ref_nodes) for i in elem],
        dtype=float,
    )
    return normalize_by_row(normals)


def compute_normal_surface_at_nodes(
    kind: VtkType, space: Mat[f64], elem: Mat[i32], LOG: ILogger = NullLogger()
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


def compute_mesh_outer_normal_at_nodes(mesh: CheartMesh, LOG: ILogger = NullLogger()):
    KIND = mesh.top.TYPE
    LOG.debug(f"{KIND.name=}")
    interp_basis = {k: KIND.shape_dfuncs(v) for k, v in enumerate(KIND.ref_nodes)}
    node_normal: dict[int, list[Vec[f64]]] = defaultdict(list)
    for elem in mesh.top.v:
        for i in range(len(interp_basis)):
            node_normal[elem[i]].append(
                compute_normal_patch(
                    interp_basis[i], mesh.space.v, elem, KIND.ref_nodes
                )
            )
    center = mesh.space.v.mean(axis=0)
    disp = normalize_by_row(mesh.space.v - center[None, :])
    normals = np.zeros_like(mesh.space.v)
    for k, node in node_normal.items():
        vals = [np.sign(v @ disp[k]) * v for v in node]
        normals[k] = sum(vals) / len(vals)
    outer = np.einsum("...i,...i", normals, disp)
    normals = normals * np.sign(outer)[:, None]
    return normalize_by_row(normals)
