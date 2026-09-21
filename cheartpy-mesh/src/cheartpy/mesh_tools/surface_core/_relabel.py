from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Err, Ok, Result

from cheartpy.mesh import CheartMesh, CheartMeshBoundary, CheartMeshPatch

if TYPE_CHECKING:
    from collections.abc import Mapping


def relabel_cheart_surfaces[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], swap: Mapping[int, int], *, closed: bool = False
) -> Result[CheartMesh[F, I]]:
    """Swap the label integer from new to old.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The mesh to relabel.
    swap : Mapping[Old: int, New: int]
        A mapping from old labels to new labels.
    closed : bool, default=False
        If True, new mesh will only contain the new labels in swap.

    Returns
    -------
    Result[CheartMesh[F, I]]
        The relabeled mesh.

    """
    if mesh.bnd is None:
        return Ok(mesh)
    if not all(b in mesh.bnd.v for b in swap):
        msg = "key in swap is not found in mesh.bnd.v."
        return Err(ValueError(msg))
    if closed:
        new_v = {
            new: CheartMeshPatch(tag=new, n=b.n, k=b.k, v=b.v, TYPE=b.TYPE)
            for old, new in swap.items()
            if (b := mesh.bnd.v[old])
        }
    else:
        new_v = {
            new: CheartMeshPatch(tag=new, n=v.n, k=v.k, v=v.v, TYPE=v.TYPE)
            for k, v in mesh.bnd.v.items()
            if (new := swap.get(k, k))
        }
    mesh.bnd = CheartMeshBoundary(n=mesh.bnd.n, v=new_v, TYPE=mesh.bnd.TYPE)
    return Ok(mesh)
