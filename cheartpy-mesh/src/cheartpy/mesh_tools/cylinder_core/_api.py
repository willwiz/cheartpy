from typing import TYPE_CHECKING, Literal, Unpack, overload

from cheartpy.mesh_tools.hex_core import create_hex_mesh
from cheartpy.mesh_tools.interpolation import create_quad_mesh_from_lin_cylindrical

from ._core import (
    convert_to_cylindrical,
    cylindrical_to_cartesian,
    merge_circ_ends,
    rotate_axis,
)
from ._types import CartesianDirection

if TYPE_CHECKING:
    import numpy as np
    from pytools.arrays import T3

    from cheartpy.mesh import CheartMesh

    from ._parsing import CylinderArgs, CylinderKwargs


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
    *,
    make_quad: Literal[False],
) -> tuple[CheartMesh[np.float64, np.intc], None]: ...
@overload
def create_cylinder_mesh(
    shape: tuple[float, float, float, float],
    dim: T3[int],
    axis: Literal["x", "y", "z"],
    *,
    make_quad: Literal[True],
) -> tuple[CheartMesh[np.float64, np.intc], CheartMesh[np.float64, np.intc]]: ...
def create_cylinder_mesh(
    shape: tuple[float, float, float, float],
    dim: T3[int],
    axis: Literal["x", "y", "z"],
    *,
    make_quad: bool = False,
):
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


def make_cylinder_api(
    args: CylinderArgs, **kwargs: Unpack[CylinderKwargs]
) -> tuple[CheartMesh[np.float64, np.intc], CheartMesh[np.float64, np.intc] | None]:
    mesh, quad = create_cylinder_mesh(
        (args["rin"], args["rout"], args["length"], args["base"]),
        (args["rn"], args["qn"], args["zn"]),
        kwargs["axis"],
        **{k: kwargs[k] for k in ["make_quad"] if k in kwargs},
    )
    if prefix := kwargs.get("prefix"):
        mesh.save(prefix)
    return mesh, quad
