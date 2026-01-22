__all__ = ["create_cylinder_mesh"]
from typing import Literal, overload

import numpy as np
from cheartpy.mesh.struct import CheartMesh
from pytools.arrays import T3

@overload
def create_cylinder_mesh(
    shape: tuple[float, float, float, float],
    dim: T3[int],
    axis: Literal["x", "y", "z"],
) -> tuple[CheartMesh[np.float64, np.intc], None]: ...
@overload
def create_cylinder_mesh(
    shape: tuple[float, float, float, float],
    dim: T3[int],
    axis: Literal["x", "y", "z"],
    make_quad: Literal[False],
) -> tuple[CheartMesh[np.float64, np.intc], None]: ...
@overload
def create_cylinder_mesh(
    shape: tuple[float, float, float, float],
    dim: T3[int],
    axis: Literal["x", "y", "z"],
    make_quad: Literal[True],
) -> tuple[CheartMesh[np.float64, np.intc], CheartMesh[np.float64, np.intc]]: ...
