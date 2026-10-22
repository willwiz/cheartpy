from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import get_boundary_element
from cheartpy.vtk import get_vtk_elem
from pytools.math import optimized_add_at
from pytools.result import Err, Ok, Result

from cheartpy.mesh_tools.tools import create_index_permutation, normalize_by_row

if TYPE_CHECKING:
    from pytools.arrays import A2, A3, Arr

    from cheartpy.mesh import CheartMesh


_REGRESS_TOL = 0.01
_DBL_TOL = 1.0e-14


def compute_normals[F: np.floating, I: np.integer](
    dbasis_x: A2[np.floating],
    dbasis_y: A2[np.floating],
    space: A2[F],
    elem: A2[I],
) -> Result[A3[F]]:
    """Vectorized computation of normals for at nodes for surface elements.

    Let:
        N = number of nodes in the element
        E = number of elements
        J = number of basis functions
        G = Points in element being evaluated at
        D = Dimension of the space (3 for 3D)

    space: [N, D]
    elem: [E, J]
    dbasis_x (basis function derivatives): [G, J]
    dbasis_y (basis function derivatives): [G, J]
    normals: [E, G, D]

    derivative in x -> nodes [E, J, D] @ dbasis_x [G, J] -> [E, G, D]
    derivative in y -> nodes [E, J, D] @ dbasis_y [G, J] -> [E, G, D]
    normals = cross(derivative in x, derivative in y) -> [E, G, D]

    Parameters
    ----------
    dbasis_x : A2[np.floating]
        The derivative of the basis functions with respect to x.
    dbasis_y : A2[np.floating]
        The derivative of the basis functions with respect to y.
    space : A2[F]
        The coordinates of the nodes in 3D space.
    elem : A2[I]
        The element connectivity.

    Returns
    -------
    A3[F]
        The computed normals for each node in each element.

    """
    nodes = space[elem]
    v1: A3[F] = np.einsum("ejd,gj->egd", nodes, dbasis_x)
    v2: A3[F] = np.einsum("ejd,gj->egd", nodes, dbasis_y)
    normals = np.cross(v1, v2)
    if not is_nonzero(normals):
        msg = "Surface has zero normals. Degenerate surface."
        return Err(ValueError(msg))
    return Ok(normals)


def compute_cell_outer[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], surf: int
) -> Result[A3[F]]:
    """Compute the outer vector for each element on the surface.

    Let:
        N = number of nodes in the element
        E = number of elements
        J = number of basis functions
        D = Dimension of the space (3 for 3D)

    space: [N, D]
    elem: [E, J]
    nodes: space[elem] -> [E, J, D]
    centroids: nodes [E, J, D] -> [E, D]
    outer: nodes [E, J, D] - centroids [E, D] -> [E, J, D]

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The mesh object containing the surface information.
    surf : int
        The surface index for which to compute the outer vectors.

    Returns
    -------
    A3[F]
        The outer vectors for each element on the specified surface.

    """
    centroids = np.mean(mesh.space.v[mesh.top.v[mesh.bnd.v[surf].k]], axis=1)  # [E, 3]
    outer = np.asarray(
        mesh.space.v[mesh.bnd.v[surf].v] - centroids[:, None, :], dtype=mesh.space.v.dtype
    )
    if not is_nonzero(outer):
        msg = f"Surface {surf} has zero outer vector. Degenerate surface."
        return Err(ValueError(msg))
    return Ok(outer)


_NORMAL_TOL = 1.0e-6


def is_nonzero[F: np.floating](vec: Arr[..., F]) -> bool:
    norm = (vec * vec).sum(axis=-1)
    return bool(np.all(norm > _NORMAL_TOL))


def orient_normals_as_outward[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], surf: int, normals: A3[F]
) -> Result[A3[F]]:
    """Orient the normals to point outward from the mesh.

    Let:
        N = number of nodes in mesh
        E = number of elements in the surface
    """
    match compute_cell_outer(mesh, surf):  # [E, 3]
        case Ok(outer): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    orientation: A2[F] = np.einsum("eji,eji->ej", normals, outer)  # [E, J]
    if not is_nonzero(orientation):
        msg = f"Surface {surf} outer vector is not aligned with normals. Degenerate surface."
        return Err(ValueError(msg))
    return Ok(normals * np.sign(orientation)[:, :, None])  # [E, J, 3]


def pack_array_to_surface_topology[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], surf: int, normals: A3[F]
) -> Result[A2[F]]:
    perm = create_index_permutation(mesh.bnd.v[surf].v)
    indicies = perm.fwd[mesh.bnd.v[surf].v]
    return Ok(optimized_add_at(indicies, normals))


def cell_average[F: np.floating](arr: A3[F]) -> Result[A2[F]]:
    cell_centered = np.mean(arr, axis=1)
    if not is_nonzero(cell_centered):
        msg = "Mean normals on surface has zero norm. Degenerate cell likely."
        return Err(ValueError(msg))
    return normalize_by_row(cell_centered).next()


def compute_surface_normal[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], in_surf: int, *, at_center: bool = False
) -> Result[A2[F]]:
    """Return the surface normal of the boundary surface `in_surf` of the mesh.

    If at_center is True, the normal is computed at the center of the surface elements. Otherwise,
    it is computed at the nodes of the surface elements.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The mesh to compute the surface normal for.
    in_surf : int
        The index of the boundary surface to compute the normal for.
    at_center : bool, optional
        Whether to compute the normal at the center of the surface elements or at the nodes.

    Returns
    -------
    Result[A2[F]]
        The surface normal of the boundary surface `in_surf` of the mesh, or an error if the mesh
        has no boundary or the surface is not found. If at_center = True, the normal is computed at
        the center of the surface elements. Otherwise, it is computed at the nodes.

    """
    if in_surf not in mesh.bnd.v:
        msg = f"Surface {in_surf} not found"
        return Err(ValueError(msg))
    surface_element = get_boundary_element(mesh.top.kind)
    vtkelement = get_vtk_elem(surface_element)
    interp_basis_at_refnodes = tuple(vtkelement.shape_dfunc(v) for v in vtkelement.ref)
    x_deriv = np.asarray([i[0, :] for i in interp_basis_at_refnodes], dtype=mesh.space.v.dtype)
    y_deriv = np.asarray([i[1, :] for i in interp_basis_at_refnodes], dtype=mesh.space.v.dtype)
    normals = (
        compute_normals(x_deriv, y_deriv, mesh.space.v, mesh.bnd.v[in_surf].v)
        .and_then(lambda n: orient_normals_as_outward(mesh, in_surf, n))
    )  # fmt: skip
    if at_center:
        return normals.and_then(cell_average)
    return (
        normals
        .and_then(lambda x: pack_array_to_surface_topology(mesh, in_surf, x))
        .and_then(normalize_by_row)
    )  # fmt: skip
