from typing import Literal

import numpy as np
from cheartpy.mesh import CheartMesh
from cheartpy.mesh_tools.tools import (
    MergedMesh,
)
from pytools.arrays import A1
from pytools.result import Result

from ._types import CLPartition

def create_centerline_partition[F: np.floating = np.float64, I: np.integer = np.intp](
    param: int | A1[F] | CLPartition[F, I],
    boundary: Literal["all", "left", "right", "none"] = "all",
) -> CLPartition[F, I]: ...
def create_centerline_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    a_z: A1[F],
    partition: int | A1[F] | CLPartition[F],
    *,
    in_surf: int | None = None,
) -> Result[MergedMesh[F, I]]: ...
