from __future__ import annotations

__all__ = ["create_hex_mesh"]
from typing import TYPE_CHECKING

from cheartpy.cheart_mesh.data import CheartMesh

from .core import (
    create_boundary,
    create_space,
    create_square_element_index,
    create_square_nodal_index,
    create_topology,
)

if TYPE_CHECKING:
    import numpy as np
    from arraystubs import T3


def create_hex_mesh(
    dim: T3[int],
    shape: T3[float] = (1.0, 1.0, 1.0),
    shift: T3[float] = (0.0, 0.0, 0.0),
) -> CheartMesh[np.float64, np.intc]:
    node_index = create_square_nodal_index(*dim)
    elem_index = create_square_element_index(*dim)
    return CheartMesh(
        create_space(shape, shift, dim, node_index),
        create_topology(*dim, node_index, elem_index),
        create_boundary(*dim, node_index, elem_index),
    )
