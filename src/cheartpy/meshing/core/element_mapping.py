import enum
from typing import Final


class ElementTypes(enum.Enum):
    HEX = (8, 27)
    TET = (4, 8)
    SQUARE = (4, 9)


SQUARE_L2Q_MAP: Final[list[list[int]]] = [
    [0],
    [1],
    [2],
    [3],
    [0, 1],
    [0, 2],
    [0, 1, 2, 3],
    [1, 3],
    [2, 3],
]

HEX_L2Q_MAP: Final[list[list[int]]] = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [0, 1],
    [0, 2],
    [0, 1, 2, 3],
    [1, 3],
    [2, 3],
    [0, 4],
    [0, 1, 4, 5],
    [1, 5],
    [0, 2, 4, 6],
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 3, 5, 7],
    [2, 6],
    [2, 3, 6, 7],
    [3, 7],
    [4, 5],
    [4, 6],
    [4, 5, 6, 7],
    [5, 7],
    [6, 7],
]


TET_L2Q_MAP: Final[list[list[int]]] = [
    [0],
    [1],
    [2],
    [3],
    [0, 1],
    [0, 2],
    [1, 2],
    [0, 3],
    [1, 3],
    [2, 3],
]


def get_elem_type(lin_size: int, quad_size: int) -> ElementTypes:
    match lin_size, quad_size:
        case ElementTypes.TET.value:
            return ElementTypes.TET
        case ElementTypes.HEX.value:
            return ElementTypes.HEX
        case ElementTypes.SQUARE.value:
            return ElementTypes.SQUARE
        case _:
            raise ValueError(f"Could not determine element type from lin = {
                             lin_size} and quad = {quad_size}")


def get_elmap(elem: ElementTypes, lin_dim: int, quad_dim: int):
    match elem, lin_dim, quad_dim:
        case ElementTypes.TET, 4, 8:
            elmap = TET_L2Q_MAP
        case ElementTypes.HEX, 8, 27:
            elmap = HEX_L2Q_MAP
        case ElementTypes.SQUARE, 4, 9:
            elmap = SQUARE_L2Q_MAP
        case _:
            raise ValueError(
                f"Mismatch in element type {elem} and either number of nodes in linear element {
                    lin_dim} or either number of nodes in quad element {quad_dim}. Should be {elem.value}."
            )
    return elmap
