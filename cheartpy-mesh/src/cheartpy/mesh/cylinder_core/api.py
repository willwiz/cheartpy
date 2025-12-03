from typing import TYPE_CHECKING, Literal

from cheartpy.mesh.hex_core.api import create_hex_mesh
from cheartpy.mesh.interpolate.remeshing import create_quad_mesh_from_lin_cylindrical

from .core import (
    convert_to_cylindrical,
    cylindrical_to_cartesian,
    merge_circ_ends,
    rotate_axis,
)
from .data import CartesianDirection

if TYPE_CHECKING:
    import numpy as np
    from cheartpy.mesh.struct import CheartMesh
    from pytools.arrays import T3


def create_cylinder_mesh(
    shape: tuple[float, float, float, float],
    dim: T3[int],
    axis: Literal["x", "y", "z"],
    *,
    make_quad: bool = False,
) -> tuple[CheartMesh[np.float64, np.intc], CheartMesh[np.float64, np.intc] | None]:
    """Create a cylindrical mesh for Cheart.

    Args:
        shape: A tuple containing the inner radius, outer radius, length, and base.
        dim: Dimensions of the mesh in the form (nx, ny, nz).
        axis: Axis around which the cylinder is oriented ('x', 'y', or 'z').
        make_quad: If True, also create a quadrilateral mesh from the cylindrical mesh.

    Returns:
        A tuple containing the cylindrical mesh and optionally a quadrilateral mesh.

    """
    r_in, r_out, length, base = shape
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
