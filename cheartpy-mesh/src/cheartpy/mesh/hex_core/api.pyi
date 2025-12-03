import numpy as np
from cheartpy.mesh.struct import CheartMesh
from pytools.arrays import T3

__all__ = ["create_hex_mesh"]

def create_hex_mesh(
    dim: T3[int],
    shape: T3[float] = (1.0, 1.0, 1.0),
    shift: T3[float] = (0.0, 0.0, 0.0),
) -> CheartMesh[np.float64, np.intc]: ...
