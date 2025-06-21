from __future__ import annotations

from typing import Literal

import numpy as np
from arraystubs import T3

from cheartpy.cheart_mesh.data import CheartMesh
from cheartpy.mesh.hex_core.api import create_hex_mesh
from cheartpy.mesh.interpolate.remeshing import create_quad_mesh_from_lin_cylindrical

from .core import (
    convert_to_cylindrical,
    cylindrical_to_cartesian,
    merge_circ_ends,
    rotate_axis,
)
from .data import CartesianDirection


def create_cylinder_mesh(
    r_in: float,
    r_out: float,
    length: float,
    base: float,
    dim: T3[int],
    axis: Literal["x", "y", "z"],
    *,
    make_quad: bool = False,
) -> tuple[CheartMesh[np.float64, np.intc], CheartMesh[np.float64, np.intc] | None]:
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
