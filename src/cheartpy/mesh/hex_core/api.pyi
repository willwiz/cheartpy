__all__ = ["create_hex_mesh"]
import numpy as np
from arraystubs import T3

from cheartpy.cheart_mesh.data import CheartMesh

def create_hex_mesh(
    dim: T3[int],
    shape: T3[float] = (1.0, 1.0, 1.0),
    shift: T3[float] = (0.0, 0.0, 0.0),
) -> CheartMesh[np.float64, np.intc]: ...
