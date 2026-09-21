import numpy as np

from cheartpy.mesh import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from cheartpy.mesh_tools.tools import create_index_permutation


def recompile_cheartmesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> CheartMesh[F, I]:
    """Recompile a CheartMesh by removing dangling nodes and fixing orientations.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The input mesh to be recompiled.

    Returns
    -------
    CheartMesh[F, I]
        The recompiled mesh with corrected orientations and no dangling nodes.

    """
    perm = create_index_permutation(mesh.top.v)
    new_x = CheartMeshSpace(mesh.space.v[perm.old])
    new_t = CheartMeshTopology(perm.fwd[mesh.top.v], mesh.top.type)
    new_bnd = CheartMeshBoundary(
        {k: CheartMeshPatch(v.tag, v.k, perm.fwd[v.v], v.type) for k, v in mesh.bnd.v.items()},
    )
    return CheartMesh(new_x, new_t, new_bnd)
