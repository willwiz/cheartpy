__all__ = ["create_hex_mesh"]
from ...var_types import *
from ...cheart_mesh import *
from .core import *


def create_hex_mesh(
    dim: T3[int],
    shape: T3[float] = (1.0, 1.0, 1.0),
    shift: T3[float] = (0.0, 0.0, 0.0),
) -> CheartMesh:
    node_index = create_square_nodal_index(*dim)
    elem_index = create_square_element_index(*dim)
    return CheartMesh(
        create_space(shape, shift, dim, node_index),
        create_topology(*dim, node_index, elem_index),
        create_boundary(*dim, node_index, elem_index),
    )
