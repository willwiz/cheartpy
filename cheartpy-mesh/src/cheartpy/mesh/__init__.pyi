# ruff: noqa: PYI011
from pathlib import Path

import numpy as np
from cheartpy.elem_interfaces import CheartEnum
from pytools.arrays import A2, DType
from pytools.result import Result

from ._struct import CheartMesh as CheartMesh
from ._struct import CheartMeshBoundary as CheartMeshBoundary
from ._struct import CheartMeshPatch as CheartMeshPatch
from ._struct import CheartMeshSpace as CheartMeshSpace
from ._struct import CheartMeshTopology as CheartMeshTopology

def cheart_mesh_from_arrays[F: np.floating, I: np.integer](
    space: A2[F], top: A2[I], bnd: A2[I] | None = None, *, elem: CheartEnum | None = None
) -> Result[CheartMesh[F, I]]: ...
def import_cheart_mesh[F: np.floating, I: np.integer](
    name: Path | str,
    forced_type: CheartEnum | None = None,
    *,
    ftype: DType[F] = np.float64,
    itype: DType[I] = np.intp,
) -> Result[CheartMesh[F, I]]: ...
