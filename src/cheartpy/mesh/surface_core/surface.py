from __future__ import annotations

__all__ = [
    "compute_mesh_outer_normal_at_nodes",
    "compute_normal_surface_at_center",
    "compute_normal_surface_at_nodes",
    "normalize_by_row",
]
from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.linalg import lstsq
from pytools.logging.api import NULL_LOGGER

from cheartpy.vtk.api import get_vtk_elem

if TYPE_CHECKING:
    from arraystubs import Arr1, Arr2
    from pytools.logging.trait import ILogger

    from cheartpy.cheart_mesh.data import CheartMesh
    from cheartpy.vtk.trait import VtkElem

_REGRESS_TOL = 0.01
_DBL_TOL = 1.0e-14


def compute_normal_patch[F: np.floating, I: np.integer](
    basis: Arr2[np.floating],
    space: Arr2[F],
    elem: Arr1[I],
    ref_space: Arr2[np.floating],
    log: ILogger = NULL_LOGGER,
) -> Arr1[F]:
    nodes = space[elem] - ref_space
    u = np.array([[nodes[:, i] @ basis[j] for j in range(3)] for i in range(3)])
    f = u + np.identity(3)
    if np.linalg.det(f) < _REGRESS_TOL:
        log.warn("Element node order is inverted.")
        f = u - np.identity(3)
    res, *_ = lstsq(f.T, np.array([0, 0, 1], dtype=float))
    return cast("Arr1[F]", res)


def normalize_by_row[F: np.floating](vals: Arr2[F]) -> Arr2[F]:
    norm = np.sqrt(np.einsum("...i,...i", vals, vals))
    norm[norm < _DBL_TOL] = 1.0
    return vals / norm[:, np.newaxis]


def compute_normal_surface_at_center[F: np.floating, I: np.integer](
    kind: VtkElem,
    space: Arr2[F],
    elem: Arr2[I],
    _log: ILogger = NULL_LOGGER,
) -> Arr2[F]:
    centroid = np.mean(kind.ref, axis=0)
    interp_basis = kind.shape_dfunc(centroid)
    normals = np.array(
        [compute_normal_patch(interp_basis, space, i, kind.ref) for i in elem],
        dtype=float,
    )
    return normalize_by_row(normals)


def compute_normal_surface_at_nodes[F: np.floating, I: np.integer](
    kind: VtkElem,
    space: Arr2[F],
    elem: Arr2[I],
    log: ILogger = NULL_LOGGER,
) -> dict[int, Arr2[F]]:
    interp_basis = {k: kind.shape_dfunc(v) for k, v in enumerate(kind.ref)}
    log.debug(f"{interp_basis=}")
    normals = {
        k: np.array(
            [compute_normal_patch(v, space, i, kind.ref) for i in elem],
            dtype=float,
        )
        for k, v in interp_basis.items()
    }
    return {k: normalize_by_row(v) for k, v in normals.items()}


def compute_mesh_outer_normal_at_nodes[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    log: ILogger = NULL_LOGGER,
) -> Arr2[F]:
    vtkelem = get_vtk_elem(mesh.top.TYPE)
    log.debug(f"{vtkelem.body=}")
    interp_basis = {k: vtkelem.shape_dfunc(v) for k, v in enumerate(vtkelem.ref)}
    node_normal: dict[int, list[Arr1[F]]] = defaultdict(list)
    for elem in mesh.top.v:
        for i in range(len(interp_basis)):
            node_normal[elem[i]].append(
                compute_normal_patch(
                    interp_basis[i],
                    mesh.space.v,
                    elem,
                    vtkelem.ref,
                ),
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
