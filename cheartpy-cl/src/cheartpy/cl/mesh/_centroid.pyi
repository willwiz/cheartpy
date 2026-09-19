from typing import Required, Unpack

import numpy as np
from cheartpy.mesh import CheartMesh
from pytools.arrays import A1, A2
from typing_extensions import TypedDict

class CentroidAPIKwargs[F: np.floating](TypedDict, total=False):
    width: float
    a_z: Required[A1[F]]
    v_z: Required[A2[F]]

def compute_a_c_coordinate[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], **kwargs: Unpack[CentroidAPIKwargs[F]]
) -> A1[F]: ...
