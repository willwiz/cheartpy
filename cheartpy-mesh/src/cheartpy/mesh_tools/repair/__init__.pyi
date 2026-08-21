import numpy as np
from pytools.result import Result

from cheartpy.mesh import CheartMesh

def fix_tetra_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> Result[CheartMesh[F, I]]: ...
