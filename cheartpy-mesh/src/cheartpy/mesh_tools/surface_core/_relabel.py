from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Ok, Result

from cheartpy.mesh import CheartMesh, CheartMeshBoundary, CheartMeshPatch

if TYPE_CHECKING:
    from collections.abc import Mapping


def relabel_cheart_surface[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], swap: Mapping[int, int]
) -> Result[CheartMesh[F, I]]:
    """Swap the label integer from new to old.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The mesh to relabel.
    swap : Mapping[Old: int, New: int]
        A mapping from old labels to new labels.

    Returns
    -------
    Result[CheartMesh[F, I]]
        The relabeled mesh.

    """
    if mesh.bnd is None:
        return mesh
    new_v = {
        swap.get(k, k): CheartMeshPatch(tag=swap.get(k, k), n=v.n, k=v.k, v=v.v, TYPE=v.TYPE)
        for k, v in mesh.bnd.v.items()
    }
    mesh.bnd = CheartMeshBoundary(n=mesh.bnd.n, v=new_v, TYPE=mesh.bnd.TYPE)
    return Ok(mesh)
