from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import get_vtk_boundary_element
from cheartpy.vtk.api import get_vtk_elem
from numpy.linalg import lstsq
from pytools.logging import get_logger
from pytools.result import Err, Ok, Result

from .meshing import create_mesh_from_surface

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cheartpy.vtk.struct import VtkElem
    from pytools.arrays import A1, A2

    from cheartpy.mesh import CheartMesh

__all__ = [
    "compute_mesh_outer_normal_at_nodes",
    "compute_surface_normal_at_nodes",
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
    # Grab the nodes of the element
    nodes = space[elem] - ref_space
    u = np.array([[nodes[:, i] @ b for b in basis] for i in range(3)])
    f = u + np.identity(3)
    if np.linalg.det(f) < _REGRESS_TOL:
        _g_log = get_logger()
        _g_log.debug("Element node order is inverted.")
        f = u - np.identity(3)
    res, *_ = lstsq(f.T, np.array([0, 0, 1], dtype=basis.dtype))
    return res.astype(space.dtype)


def normalize_by_row[F: np.floating](vals: A2[F]) -> A2[F]:
    norm = np.sqrt(np.einsum("...i,...i", vals, vals))
    # norm[norm < _DBL_TOL] = 1.0
    return vals / norm[:, np.newaxis]


def compute_surface_normal_at_center[F: np.floating, I: np.integer](
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


def compute_surface_normal_at_nodes[F: np.floating, I: np.integer](
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


def orient_normals_as_outward[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], in_surf: int, normals: Mapping[I, Mapping[I, A1[F]]]
) -> Result[Mapping[I, Mapping[I, A1[F]]]]:
    if mesh.bnd is None:
        msg = "Mesh has no boundary"
        return Err(ValueError(msg))
    if in_surf not in mesh.bnd.v:
        msg = f"Surface {in_surf} not found"
        return Err(ValueError(msg))
    surf_elem = get_vtk_boundary_element(mesh.top.TYPE)
    if surf_elem is None:
        msg = f"Unsupported mesh type: {mesh.top.TYPE}"
        return Err(ValueError(msg))
    bnd_elem = {k: mesh.top.v[k] for k in mesh.bnd.v[in_surf].k}
    bnd_elem_centroids = {k: mesh.space.v[elem].mean(axis=0) for k, elem in bnd_elem.items()}
    bnd_patch_outer = {
        k: {b: mesh.space.v[b] - bnd_elem_centroids[k] for b in bnd}
        for k, bnd in zip(mesh.bnd.v[in_surf].k, mesh.bnd.v[in_surf].v, strict=True)
    }
    fix_direction = {
        k: {b: np.sign(n.dot(bnd_patch_outer[k][b])) * n for b, n in surf.items()}
        for k, surf in normals.items()
    }
    return Ok(fix_direction)


def pack_array_to_surface_topology[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], in_surf: int, dct_values: Mapping[I, Mapping[I, A1[F]]]
) -> Result[A2[F]]:
    match create_mesh_from_surface(mesh, in_surf):
        case Ok(surf_mesh): ...  # fmt: skip
        case Err(e):
            return Err(e)
    mesh_bnd = mesh.bnd
    if mesh_bnd is None:
        msg = "Mesh has no boundary"
        return Err(ValueError(msg))
    node_map = {
        i: j
        for old, new in zip(mesh_bnd.v[in_surf].v, surf_mesh.top.v, strict=True)
        for i, j in zip(old, new, strict=True)
    }
    normal_array = np.zeros_like(surf_mesh.space.v)
    for ns in dct_values.values():
        for b, n in ns.items():
            normal_array[node_map[b]] += n
    return Ok(normalize_by_row(normal_array))


def compute_surface_normal[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], in_surf: int
) -> Result[A2[F]]:
    if mesh.bnd is None:
        msg = "Mesh has no boundary"
        return Err(ValueError(msg))
    if in_surf not in mesh.bnd.v:
        msg = f"Surface {in_surf} not found"
        return Err(ValueError(msg))
    surf_elem = get_vtk_boundary_element(mesh.top.TYPE)
    if surf_elem is None:
        msg = f"Unsupported mesh type: {mesh.top.TYPE}"
        return Err(ValueError(msg))
    vtkelem = get_vtk_elem(surf_elem)
    interp_basis_at_refnodes = tuple(vtkelem.shape_dfunc(v) for v in vtkelem.ref)
    bnd_patches: dict[I, A1[I]] = dict(
        zip(mesh.bnd.v[in_surf].k, mesh.bnd.v[in_surf].v, strict=True)
    )
    normals = {
        k: {
            b: compute_normal_patch(basis, mesh.space.v, patch, vtkelem.ref)
            for b, basis in zip(patch, interp_basis_at_refnodes, strict=True)
        }
        for k, patch in bnd_patches.items()
    }
    match orient_normals_as_outward(mesh, in_surf, normals):
        case Ok(normals): ...  # fmt: skip
        case Err(e):
            return Err(e)
    match pack_array_to_surface_topology(mesh, in_surf, normals):
        case Ok(normal_array): ...  # fmt: skip
        case Err(e):
            return Err(e)
    return Ok(normalize_by_row(normal_array))
