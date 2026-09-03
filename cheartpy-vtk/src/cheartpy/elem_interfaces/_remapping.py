from typing import TYPE_CHECKING

import numpy as np

from ._interface import convert_element_type, get_node_order
from ._types import AbaqusEnum, CheartEnum, ElemEnum, ElemType, GmshEnum, VtkEnum

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A1

_Enum: Mapping[ElemType, type[ElemEnum]] = {
    "Cheart": CheartEnum,
    "Vtk": VtkEnum,
    "Abaqus": AbaqusEnum,
    "Gmsh": GmshEnum,
}


def get_node_permutation(input_type: ElemEnum, target: ElemType) -> A1[np.intp]:
    """Return the permutation array to convert the node ordering of an element type to another.

    Parameters
    ----------
    input_type : CheartEnum | VtkEnum | AbaqusEnum | GmshEnum
        The input element enum.
    target : Literal["Cheart", "Vtk", "Abaqus", "Gmsh"]
        The target element class string.

    Returns
    -------
    A1[np.intp]
        The permutation array to convert the node ordering of an element type to another.

    """
    target_type = convert_element_type(input_type, target)
    input_order = get_node_order(input_type)
    target_order = get_node_order(target_type)
    mapping = {v: o for o, v in target_order.items()}
    perm_dct = {i: mapping[v] for i, v in input_order.items()}
    perm = np.zeros(len(input_order), dtype=np.intp)
    perm[list(perm_dct.values())] = list(perm_dct.keys())
    return perm
