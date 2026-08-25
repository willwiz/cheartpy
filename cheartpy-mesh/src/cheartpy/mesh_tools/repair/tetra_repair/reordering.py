from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import VtkEnum
from pytools.result import Err, Ok, Result

from cheartpy.mesh import CheartMesh, CheartMeshBoundary, CheartMeshPatch, CheartMeshTopology

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A1, A2


def fix_negative_volume[F: np.floating, I: np.integer](nodes: A2[F], connectivity: A2[I]) -> A2[I]:
    """Fix the orientation of tetrahedral elements to ensure positive volume.

    Parameters
    ----------
    nodes : A2[F]
        The coordinates of the nodes in the mesh.
    connectivity : A2[I]
        The connectivity of the tetrahedral elements.

    Returns
    -------
    A2[I]
        The corrected connectivity with positive volume.

    """
    # Calculate the volume of each tetrahedron
    v0 = nodes[connectivity[:, 0]]
    v1 = nodes[connectivity[:, 1]]
    v2 = nodes[connectivity[:, 2]]
    v3 = nodes[connectivity[:, 3]]

    # Compute the signed volume using the scalar triple product
    volumes = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0)

    # Identify elements with negative volume
    negative_volume_indices = np.where(volumes < 0)[0]
    if len(negative_volume_indices) == 0:
        print("No negative volume.")
        return connectivity
    print(
        "negative volume found for",
        len(negative_volume_indices),
        " of ",
        len(connectivity),
        "elements.",
    )
    # Swap two vertices to correct the orientation for negative volume elements
    connectivity[negative_volume_indices] = connectivity[negative_volume_indices][:, [0, 1, 3, 2]]

    return connectivity


def fix_boundary_orientation[F: np.floating, I: np.integer](
    nodes: A2[F], connectivity: A2[I], bnd: tuple[A1[I], A2[I]]
) -> A2[I]:
    """Fix the orientation of boundary faces to ensure consistent outward normals.

    Parameters
    ----------
    nodes : A2[F]
        The coordinates of the nodes in the mesh.
    connectivity : A2[I]
        The connectivity of the tetrahedral elements.
    bnd : tuple[A1[I], A2[I]]
        A tuple containing the indices of the element their belong to and the connectivity of the
        boundary faces.

    Returns
    -------
    A2[I]
        The corrected connectivity with consistent orientation.

    """
    elem, patch = bnd
    centroid = np.mean(nodes[connectivity[elem]], axis=1)
    face_centroid = np.mean(nodes[patch], axis=1)
    out_vector = face_centroid - centroid
    face_normal = np.cross(
        nodes[patch[:, 1]] - nodes[patch[:, 0]], nodes[patch[:, 2]] - nodes[patch[:, 0]]
    )
    dot_product = np.einsum("ij,ij->i", face_normal, out_vector)
    negative_orientation_indices = np.where(dot_product < 0)[0]
    if len(negative_orientation_indices) == 0:
        print("No boundary faces with negative orientation.")
        return patch
    print(
        "negative orientation found for",
        len(negative_orientation_indices),
        "of",
        len(patch),
        "faces.",
    )
    patch[negative_orientation_indices] = patch[negative_orientation_indices][:, [0, 2, 1]]
    return patch


def reorient_tetra_boundary[F: np.floating, I: np.integer](
    nodes: A2[F], connectivity: A2[I], bnd: CheartMeshBoundary[I] | None
) -> CheartMeshBoundary[I] | None:
    """Reorient the boundary faces of a tetrahedral mesh to ensure consistent outward normals.

    Parameters
    ----------
    nodes : A2[F]
        The coordinates of the nodes in the mesh.
    connectivity : A2[I]
        The connectivity of the tetrahedral elements.
    bnd : CheartMeshBoundary
        The boundary information of the mesh.

    Returns
    -------
    CheartMeshBoundary
        The corrected boundary information with consistent orientation.

    """
    if bnd is None:
        return None

    new_patches: Mapping[int, A2[I]] = {
        k: fix_boundary_orientation(nodes, connectivity, (v.k, v.v)) for k, v in bnd.v.items()
    }
    return CheartMeshBoundary(
        bnd.n,
        {k: CheartMeshPatch(v.tag, v.n, v.k, new_patches[k], v.TYPE) for k, v in bnd.v.items()},
        bnd.TYPE,
    )


def fix_tetra_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> Result[CheartMesh[F, I]]:
    """Fix the orientation of tetrahedral elements and boundary faces in a mesh.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The input mesh to be fixed.

    Returns
    -------
    CheartMesh[F, I]
        The corrected mesh with positive volume and consistent boundary orientation.

    """
    print("Trying to fix tetrahedral mesh.")
    if mesh.top.TYPE is not VtkEnum.TETRAHEDRON1:
        msg = f"Unsupported element type: {mesh.top.TYPE}. Only TETRAHEDRON1 is supported."
        return Err(ValueError(msg))
    new_connectivity = fix_negative_volume(mesh.space.v, mesh.top.v)
    new_top = CheartMeshTopology(mesh.top.n, new_connectivity, mesh.top.TYPE)
    new_bnd = (
        mesh.bnd
        if mesh.bnd is None
        else reorient_tetra_boundary(mesh.space.v, new_connectivity, mesh.bnd)
    )
    return Ok(CheartMesh(mesh.space, new_top, new_bnd))
