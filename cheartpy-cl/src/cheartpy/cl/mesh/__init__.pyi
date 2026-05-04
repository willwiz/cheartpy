from typing import Unpack

import numpy as np
from cheartpy.mesh import CheartMesh
from pytools.arrays import A1, DType
from pytools.result import Result

from ._centroid import CentroidAPIKwargs as CentroidAPIKwargs
from ._types import APIKwargs as APIKwargs
from ._types import CLDef as CLDef
from ._types import CLMesh as CLMesh
from ._types import CLPartition as CLPartition
from ._types import CLPrefix as CLPrefix
from ._types import CLSegmentDef as CLSegmentDef
from ._types import CLVectorDef as CLVectorDef

def get_cl_ftype[F: np.floating = np.float64](defn: CLDef[F]) -> DType[F]: ...
def create_cl_partition[F: np.floating](defn: CLDef[F]) -> CLPartition[F]: ...
def create_centerline_topology[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], defn: CLDef[F], **kwargs: Unpack[APIKwargs]
) -> Result[CLMesh[F, I]]: ...
def create_centerline_topology_in_surf[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], in_surf: int, defn: CLDef[F], **kwargs: Unpack[APIKwargs]
) -> Result[CLMesh[F, I]]: ...
def export_cl_mesh[F: np.floating, I: np.integer](mesh: CLMesh[F, I], defn: CLDef[F]) -> None: ...
def compute_a_c_coordinate[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], **kwargs: Unpack[CentroidAPIKwargs[F]]
) -> A1[F]: ...
