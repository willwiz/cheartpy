from collections import defaultdict
from typing import cast
import numpy as np
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
    LOG: _Logger = NullLogger(),
) -> Vec[f64]:
    # nodes = space[elem]
    # F = np.identity(3)
    # scaling = np.array(
    #     [
    #         [0.01, 0.0, 0.0],
    #         [0.0, 0.01, 0.0],
    #         [0.0, 0.0, 1.0],
    #     ],
    #     dtype=float,
    # )
    # for _ in range(1, 40):
    #     scaling = np.diag([2, 2, 1]) @ scaling
    #     nodes = nodes - ref_space @ scaling
    #     F = np.array(
    #         [[nodes[:, i] @ basis[j] for j in range(3)] for i in range(3)]
    #     ) @ np.linalg.inv(scaling) + np.identity(3)
    #     if np.abs(np.linalg.det(F)) > 0.5:
    #         break
    # else:
    #     print("algorithm failed")
    nodes = space[elem] - ref_space
    U = np.array([[nodes[:, i] @ basis[j] for j in range(3)] for i in range(3)])
    F = U + np.identity(3)
    if np.linalg.det(F) < 0.01:
        LOG.warn(f"Element node order is inverted.")
        F = U - np.identity(3)
    # if np.abs(np.linalg.det(F)) < 1e-6:
    #     print(
    #         np.linalg.det(F),
    #         F,
    #         np.array([[nodes[:, i] @ basis[j] for j in range(3)] for i in range(3)]),
    #         basis,
    #     )
    #     print(space[elem])
    # F = F / np.abs(np.linalg.det(F))
    res, *_ = lstsq(F.T, np.array([0, 0, 1], dtype=float), lapack_driver="gelsy")
    return res


def normalize_by_row(vals: Mat[f64]) -> Mat[f64]:
    norm = np.sqrt(np.einsum("...i,...i", vals, vals))
    return vals / norm[:, np.newaxis]


def compute_normal_surface_at_center(
    kind: VtkType, space: Mat[f64], elem: Mat[i32], LOG: _Logger = NullLogger()
):
    centroid = np.mean(kind.ref_nodes, axis=0)
    interp_basis = kind.shape_dfuncs(centroid)
    normals = np.array(
        [compute_normal_patch(interp_basis, space, i, kind.ref_nodes) for i in elem],
        dtype=float,
    )
    # LOG.debug(f"{normals=}")
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


def compute_mesh_normal_at_nodes(mesh: CheartMesh, LOG: _Logger = NullLogger()):
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
        norms = abs(normals[k] @ disp[k])
        # if norms < 0.2:
        #     print(k, norms, len(vals))
        #     print(f"{node=}")
    outer = np.einsum("...i,...i", normals, disp)
    normals = normals * np.sign(outer)[:, None]
    return normalize_by_row(normals)
