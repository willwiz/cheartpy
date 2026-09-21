from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import CheartEnum
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from cheartpy.mesh import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)

if TYPE_CHECKING:
    from pytools.arrays import A1, A2


def permute_reverse_cuthill_mckee[I: np.integer](connectivity: A2[I]) -> A1[I]:
    """Compute new node ordering using the Reverse Cuthill-McKee algorithm.

    Improve FE matrix bandwidth by reordering nodes based on connectivity.

    Parameters
    ----------
    connectivity : A2[I]
        2D array of shape (n_elements, n_nodes_per_element) representing the connectivity of the
        mesh.

    Returns
    -------
    A1[I]
        1D array of shape (n_nodes,) representing the new node ordering, where new = old[perm]

    """
    nnodes = connectivity.max() + 1
    rows = np.fromiter(
        (i for elem in connectivity for i in elem for j in elem if i != j), dtype=connectivity.dtype
    )
    cols = np.fromiter(
        (j for elem in connectivity for i in elem for j in elem if i != j), dtype=connectivity.dtype
    )
    data = np.ones_like(rows, dtype=rows.dtype)
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(nnodes, nnodes))
    return np.asarray(reverse_cuthill_mckee(adj_matrix, symmetric_mode=True), connectivity.dtype)


def reorder_cheartmesh[F: np.floating, I: np.integer, T: CheartEnum, B: CheartEnum](
    mesh: CheartMesh[F, I, T, B],
) -> CheartMesh[F, I, T, B]:
    """Reorder mesh nodes and connectivity using the Reverse Cuthill-McKee algorithm.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The input mesh to be reordered.

    Returns
    -------
    CheartMesh[F, I]
        The reordered mesh with updated node ordering and connectivity.

    """
    perm_rev = permute_reverse_cuthill_mckee(mesh.top.v)
    perm_fwd = np.empty_like(perm_rev)
    perm_fwd[perm_rev] = np.arange(len(perm_rev), dtype=perm_rev.dtype)
    new_space = CheartMeshSpace(mesh.space.v[perm_rev])
    new_top = CheartMeshTopology(perm_fwd[mesh.top.v], mesh.top.type)
    new_bnd = CheartMeshBoundary(
        {k: CheartMeshPatch(v.tag, v.k, perm_fwd[v.v], v.type) for k, v in mesh.bnd.v.items()},
    )
    return CheartMesh(new_space, new_top, new_bnd)
