from typing import TYPE_CHECKING, Any, TypeIs

import numpy as np

from cheartpy.mesh import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)

from ._types import IndexPermutation

if TYPE_CHECKING:
    from pytools.arrays import A1, A2


def _id_1d[I: np.integer](arr: np.ndarray[Any, np.dtype[I]]) -> TypeIs[A1[I]]:
    return arr.ndim == 1


def create_index_permutation[I: np.integer](
    index: A1[I] | A2[I], first: int = 0
) -> IndexPermutation[I]:
    """Create a permutation mapping from an index array.

    If the input is a 1D array, it is assumed to be built from node index.
    If the input is a 2D array, it is assumed to be built from topological connectivity.
    If 2D, the inv field should be used to update the nodes, i.e.,
        `new_nodes = old_nodes[perm.inv]`.
    The fwd field should be used to update the topological connectivity, i.e.,
        `new_top = perm.fwd[old_top]`.

    Parameters
    ----------
    index : A1[I] | A2[I]
        Can be a 1D array of the index of the nodes
        or a 2D array containing the topological connectivity.

    first : int, default=0
        The starting index for the forward permutation.

    Returns
    -------
    IndexPermutation[I]
        A dataclass containing the forward and inverse permutation arrays.

    """
    new = np.arange(first, len(np.unique(index.flatten())) + first, dtype=index.dtype)
    old = index if _id_1d(index) else np.unique(index.flatten())
    perm_fwd = np.full(np.max(old) + 1, -1, dtype=old.dtype)
    perm_fwd[old] = new
    return IndexPermutation(idx=old, fwd=perm_fwd)


def recompile_cheart_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> CheartMesh[F, I]:
    """Recompile a Cheart mesh to ensure that the node indices are contiguous and start from 0.

    Parameters
    ----------
    mesh
        The Cheart mesh to recompile.

    Returns [Result object]
    -----------------------
    CheartMesh[F, I]
        Recompiled Cheart mesh.

    """
    perm = create_index_permutation(mesh.top.v)
    new_x = mesh.space.v[perm.idx]
    new_t = perm.fwd[mesh.top.v]
    boundary = (
        CheartMeshBoundary(
            mesh.bnd.n,
            {
                k: CheartMeshPatch(v.tag, v.n, v.k, perm.fwd[v.v], v.TYPE)
                for k, v in mesh.bnd.v.items()
            },
            mesh.bnd.TYPE,
        )
        if mesh.bnd
        else None
    )
    return CheartMesh(
        CheartMeshSpace(len(new_x), new_x),
        CheartMeshTopology(len(new_t), new_t, mesh.top.TYPE),
        boundary,
    )
