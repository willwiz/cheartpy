from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import get_vtk_boundary_element
from cheartpy.vtk.api import get_vtk_elem
from numpy.linalg import lstsq
from pytools.logging import get_logger
from pytools.result import Err, Ok, Result, all_ok

from cheartpy.mesh_tools.tools import normalize_by_row

from .meshing import create_mesh_from_surface

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cheartpy.vtk.struct import VtkElem
    from pytools.arrays import A1, A2

    from cheartpy.mesh import CheartMesh

__all__ = [
    "compute_mesh_outer_normal_at_nodes",
    "compute_surface_normal_at_nodes",
]

_REGRESS_TOL = 0.01
_DBL_TOL = 1.0e-14


def compute_patch_normal[F: np.floating, I: np.integer](
    basis: A2[np.floating],
    space: A2[F],
    elem: A1[I],
    _ref_space: A2[np.floating],
) -> A1[F]:
    v1 = np.asarray([space[elem][:, i] @ basis[0] for i in range(3)])
    v2 = np.asarray([space[elem][:, i] @ basis[1] for i in range(3)])
    return np.cross(v1, v2).astype(space.dtype)


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
    print(basis)
    print(nodes)
    print(f)
    if np.linalg.det(f) < _REGRESS_TOL:
        _g_log = get_logger()
        _g_log.debug("Element node order is inverted.")
        f = u - np.identity(3)
        print(f)
    res, *_ = lstsq(f.T, np.array([0, 0, 1], dtype=basis.dtype))
    return res.astype(space.dtype)


def compute_surface_normal_at_center[F: np.floating, I: np.integer](
    kind: VtkElem,
    space: A2[F],
    elem: A2[I],
) -> Result[A2[F]]:
    centroid = np.mean(kind.ref, axis=0)
    interp_basis = kind.shape_dfunc(centroid)
    normals = np.array(
        [compute_normal_patch(interp_basis, space, i, kind.ref) for i in elem],
        dtype=space.dtype,
    )
    return normalize_by_row(normals).next()


def compute_surface_normal_at_nodes[F: np.floating, I: np.integer](
    kind: VtkElem,
    space: A2[F],
    elem: A2[I],
) -> Result[Mapping[int, A2[F]]]:
    interp_basis = {k: kind.shape_dfunc(v) for k, v in enumerate(kind.ref)}
    normals = {
        k: np.array(
            [compute_normal_patch(v, space, i, kind.ref) for i in elem],
            dtype=space.dtype,
        )
        for k, v in interp_basis.items()
    }
    return all_ok({k: normalize_by_row(v) for k, v in normals.items()}).next()


def compute_mesh_outer_normal_at_nodes[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> Result[A2[F]]:
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
    match normalize_by_row(mesh.space.v - center[None, :]):
        case Ok(disp): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    normals = np.zeros_like(mesh.space.v)
    for k, node in node_normal.items():
        vals = [np.sign(v @ disp[k]) * v for v in node]
        normals[k] = sum(vals) / len(vals)
    outer = np.einsum("...i,...i", normals, disp)
    normals = normals * np.sign(outer)[:, None]
    return normalize_by_row(normals).next()


def is_nonzero[F: np.floating](vec: A1[F]) -> bool:
    return np.linalg.norm(vec) > 0.0


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
    for k, v in bnd_elem_centroids.items():
        if not is_nonzero(v):
            msg_0 = f"Boundary patch {k} centroid is degenerate."
            raise ValueError(msg_0)
    for k, v in bnd_patch_outer.items():
        for b, n in v.items():
            if not is_nonzero(n):
                msg_0 = f"Boundary patch {k} {b} = {n} is degenerate."
                print(msg_0)

    fix_direction = {
        k: {b: np.sign(n.dot(bnd_patch_outer[k][b])) * n for b, n in surf.items()}
        for k, surf in normals.items()
    }
    for k, v in fix_direction.items():
        for b, n in v.items():
            if not is_nonzero(n):
                msg_0 = f"Boundary patch {k} {b} reoriented normal = {n} is degenerate."
                raise ValueError(msg_0)
    return Ok(fix_direction)


def pack_array_to_surface_topology[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], in_surf: int, dct_values: Mapping[I, Mapping[I, A1[F]]]
) -> Result[A2[F]]:
    match create_mesh_from_surface(mesh, in_surf):
        case Ok(surf_mesh): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
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
    return normalize_by_row(normal_array).next()


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
            b: compute_patch_normal(basis, mesh.space.v, patch, vtkelem.ref)
            for b, basis in zip(patch, interp_basis_at_refnodes, strict=True)
        }
        for k, patch in bnd_patches.items()
    }
    match orient_normals_as_outward(mesh, in_surf, normals):
        case Ok(normals): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    match pack_array_to_surface_topology(mesh, in_surf, normals):
        case Ok(normal_array): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    return normalize_by_row(normal_array).next()
