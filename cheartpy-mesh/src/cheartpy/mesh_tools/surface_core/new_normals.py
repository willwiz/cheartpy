from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from pytools.arrays import A1, A2, A3, Arr, SAny

    from cheartpy.mesh import CheartMesh


_REGRESS_TOL = 0.01
_DBL_TOL = 1.0e-14


def compute_normals[F: np.floating, I: np.integer](
    dbasis_x: A2[np.floating],
    dbasis_y: A2[np.floating],
    space: A2[F],
    elem: A2[I],
) -> A3[F]:
    """Vectorized computation of normals for at nodes for surface elements.

    Let:
        N = number of nodes in the element
        E = number of elements
        J = number of basis functions
        L = number of nodes to be evaluated at

    space: [N, 3]
        => The coordinates of the nodes in 3D space.
    elem: [E, J]
        => Element
        => Node in Element
    dbasis_x (basis function derivatives): [L, J]
        => Node
        => Basis function
    dbasis_y (basis function derivatives): [L, J]
        => Node
        => Basis function

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
    nodes = space[elem]  # [E, J, 3]
    v1: A3[F] = np.einsum("eji,lj->eli", nodes, dbasis_x)  # [E, J, 3]
    v2: A3[F] = np.einsum("eji,lj->eli", nodes, dbasis_y)  # [E, J, 3]
    return np.cross(v1, v2)


def compute_cell_outer[F: np.floating, I: np.integer](mesh: CheartMesh[F, I], surf: int) -> A2[F]:
    """Compute the outer vector for each element on the surface.

    Let:
        N = number of nodes in mesh
        E = number of elements in the surface
    """
    centroids = np.mean(mesh.space.v[mesh.top.v[mesh.bnd.v[surf].k]], axis=1)  # [E, 3]
    centers = np.mean(mesh.space.v[mesh.bnd.v[surf].v], axis=1)  # [E, 3]
    return np.asarray(centers - centroids, dtype=mesh.space.v.dtype)


def orient_normals_as_outward[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], surf: int, normals: A3[F]
) -> Result[A3[F]]:
    """Orient the normals to point outward from the mesh.

    Let:
        N = number of nodes in mesh
        E = number of elements in the surface
    """
    if surf not in mesh.bnd.v:
        msg = f"Surface {surf} not found"
        return Err(ValueError(msg))
    outer = compute_cell_outer(mesh, surf)  # [E, 3]
    orientation = np.einsum("eji,ei->ej", normals, outer)  # [E, J]
    normals = normals * np.sign(orientation)[:, :, None]  # [E, J, 3]
    return Ok(normals)


def optimized_add_at[F: np.floating](index: Arr[SAny, np.integer], weights: Arr[SAny, F]) -> A1[F]:
    """Optimized version of np.add.at for 1D arrays.

    Parameters
    ----------
    index : Arr[SAny, np.integer]
        The indices at which to add the weights.
    weights : Arr[SAny, T]
        The weights to be added.

    Returns
    -------
    A1[T]
        The resulting array after adding the weights at the specified indices.

    """
    size = int(index.max() + 1)
    columns = [
        np.bincount(index.ravel(), weights=weights[:, i].ravel(), minlength=size)
        for i in range(weights.shape[-1])
    ]
    return np.asarray(np.column_stack(columns), dtype=weights.dtype)
