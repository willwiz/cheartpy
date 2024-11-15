__all__ = ["create_cylinder_mesh"]
from typing import Literal, overload
from ...cheart_mesh import *
from ...var_types import *

@overload
def create_cylinder_mesh(
    r_in: float,
    r_out: float,
    length: float,
    base: float,
    dim: V3[int],
    axis: Literal["x", "y", "z"],
    make_quad: Literal[False],
) -> tuple[CheartMesh, None]: ...
@overload
def create_cylinder_mesh(
    r_in: float,
    r_out: float,
    length: float,
    base: float,
    dim: V3[int],
    axis: Literal["x", "y", "z"],
    make_quad: Literal[True],
) -> tuple[CheartMesh, CheartMesh]: ...
