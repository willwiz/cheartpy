from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np
from cheartpy.vtk.api import get_vtk_elem
from numpy.linalg import lstsq
from pytools.logging import get_logger

if TYPE_CHECKING:
    from cheartpy.mesh.struct import CheartMesh
    from cheartpy.vtk.types import VtkElem
    from pytools.arrays import A1, A2

__all__ = [
    "compute_mesh_outer_normal_at_nodes",
    "compute_normal_surface_at_center",
    "compute_normal_surface_at_nodes",
    "normalize_by_row",
]

_REGRESS_TOL = 0.01
_DBL_TOL = 1.0e-14


def compute_normal_patch[F: np.floating, I: np.integer](
    basis: A2[np.floating],
    space: A2[F],
    elem: A1[I],
    ref_space: A2[np.floating],
) -> A1[F]:
    nodes = space[elem] - ref_space
    u = np.array([[nodes[:, i] @ b for b in basis] for i in range(3)])
    f = u + np.identity(3)
    if np.linalg.det(f) < _REGRESS_TOL:
        _g_log = get_logger()
        _g_log.debug("Element node order is inverted.")
        f = u - np.identity(3)
    res, *_ = lstsq(f.T, np.array([0, 0, 1], dtype=float))
    return cast("A1[F]", res)


def normalize_by_row[F: np.floating](vals: A2[F]) -> A2[F]:
    norm = np.sqrt(np.einsum("...i,...i", vals, vals))
    # norm[norm < _DBL_TOL] = 1.0
    return vals / norm[:, np.newaxis]


def compute_normal_surface_at_center[F: np.floating, I: np.integer](
    kind: VtkElem,
    space: A2[F],
    elem: A2[I],
) -> A2[F]:
    centroid = np.mean(kind.ref, axis=0)
    interp_basis = kind.shape_dfunc(centroid)
    normals = np.array(
        [compute_normal_patch(interp_basis, space, i, kind.ref) for i in elem],
        dtype=space.dtype,
    )
    return normalize_by_row(normals)


def compute_normal_surface_at_nodes[F: np.floating, I: np.integer](
    kind: VtkElem,
    space: A2[F],
    elem: A2[I],
) -> dict[int, A2[F]]:
    interp_basis = {k: kind.shape_dfunc(v) for k, v in enumerate(kind.ref)}
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
) -> A2[F]:
    vtkelem = get_vtk_elem(mesh.top.TYPE)
    interp_basis = {k: vtkelem.shape_dfunc(v) for k, v in enumerate(vtkelem.ref)}
    node_normal: dict[int, list[A1[F]]] = defaultdict(list)
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
