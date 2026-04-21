from pathlib import Path
from typing import TypedDict, Unpack

import numpy as np
from cheartpy.cl.types import CLDef
from cheartpy.mesh import CheartMesh
from pytools.result import Result

from ._types import APIKwargs as APIKwargs
from ._types import CLMesh as CLMesh
from ._types import CLPartition as CLPartition

class _ExportKwargs(TypedDict, total=False):
    root: Path

def create_cl_partition[F: np.floating](defn: CLDef[F]) -> CLPartition[F]: ...
def create_centerline_topology[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], defn: CLDef[F], **kwargs: Unpack[APIKwargs[F]]
) -> Result[CLMesh[F, I]]: ...
def create_centerline_topology_in_surf[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], in_surf: int, defn: CLDef[F], **kwargs: Unpack[APIKwargs[F]]
) -> Result[CLMesh[F, I]]: ...
def export_cl_mesh[F: np.floating, I: np.integer](
    mesh: CLMesh[F, I], prefix: str, **kwargs: Unpack[_ExportKwargs]
) -> None: ...
