from typing import TYPE_CHECKING, Unpack

from cheartpy.mesh import CheartMesh

from ._core import (
    create_boundary,
    create_space,
    create_square_element_index,
    create_square_nodal_index,
    create_topology,
)

if TYPE_CHECKING:
    import numpy as np
    from pytools.arrays import T3, ToFloat, ToInt

    from ._parsing import BlockArgs, BlockKwargs

__all__ = ["create_hex_mesh"]


def create_hex_mesh(
    dim: T3[ToInt],
    shape: T3[ToFloat] = (1.0, 1.0, 1.0),
    offset: T3[ToFloat] = (0.0, 0.0, 0.0),
) -> CheartMesh[np.float64, np.intc]:
    node_index = create_square_nodal_index(*dim)
    elem_index = create_square_element_index(*dim)
    return CheartMesh(
        create_space(shape, offset, dim, node_index),
        create_topology(*dim, node_index, elem_index),
        create_boundary(*dim, node_index, elem_index),
    )


def make_block_cli(args: BlockArgs, **kwargs: Unpack[BlockKwargs]) -> None:
    mesh = create_hex_mesh((args["xn"], args["yn"], args["zn"]), **kwargs)
    mesh.save(args["prefix"])
