from typing import Literal
from ...cheart_mesh import *
from ...var_types import *
from ..hex_core import create_hex_mesh
from ..interpolate import create_quad_mesh_from_lin_cylindrical
from .core import *
from .data import *


def create_cylinder_mesh(
    r_in: float,
    r_out: float,
    length: float,
    base: float,
    dim: T3[int],
    axis: Literal["x", "y", "z"],
    make_quad: bool = False,
) -> tuple[CheartMesh, CheartMesh | None]:
    cube = create_hex_mesh(dim)
    cylinder = convert_to_cylindrical(cube, r_in, r_out, length, base)
    cylinder = merge_circ_ends(cylinder)
    g = cylindrical_to_cartesian(cylinder)
    g = rotate_axis(g, CartesianDirection[axis])
    if make_quad:
        quad = create_quad_mesh_from_lin_cylindrical(cylinder)
        quad = cylindrical_to_cartesian(quad)
        quad = rotate_axis(quad, CartesianDirection[axis])
        return g, quad
    return g, None
