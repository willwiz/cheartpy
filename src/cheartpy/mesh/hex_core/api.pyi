__all__ = ["create_hex_mesh"]
from ...var_types import *
from ...cheart_mesh import *

def create_hex_mesh(
    dim: T3[int],
    shape: T3[float] = (1.0, 1.0, 1.0),
    shift: T3[float] = (0.0, 0.0, 0.0),
) -> CheartMesh: ...
