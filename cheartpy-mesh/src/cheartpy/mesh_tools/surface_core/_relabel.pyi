from collections.abc import Mapping

import numpy as np
from pytools.result import Result

from cheartpy.mesh import CheartMesh

def relabel_cheart_surfaces[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], swap: Mapping[int, int]
) -> Result[CheartMesh[F, I]]: ...
