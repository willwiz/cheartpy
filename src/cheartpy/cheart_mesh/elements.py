__all__ = [
    "VTK_ELEM_TYPE",
    "VTK_ELEM",
    "VtkType",
    "guess_elem_type_from_dim",
]
import enum
import numpy as np
import dataclasses as dc
from typing import Callable, Final, Literal, Mapping, TextIO
from ..var_types import *
from .shape_functions import *

VTK_ELEM_TYPE = Literal[
    "VtkLineLinear",
    "VtkLineQuadratic",
    "VtkTriangleLinear",
    "VtkTriangleQuadratic",
    "VtkQuadrilateralLinear",
    "VtkQuadrilateralQuadratic",
    "VtkTetrahedronLinear",
    "VtkTetrahedronQuadratic",
    "VtkHexahedronLinear",
    "VtkHexahedronQuadratic",
]


@dc.dataclass(slots=True)
class VtkElemInterface:
    elem: VTK_ELEM_TYPE
    surf: VTK_ELEM_TYPE | None
    vtkelementid: Final[int]
    vtksurfaceid: Final[int | None]
    nodeordering: Final[tuple[int, ...]]
    connectivity: Final[tuple[int, ...]]
    ref_order: Final[Mat[i32]]
    ref_nodes: Final[Mat[f64]]
    shape_funcs: Callable[[Vec[f64]], Vec[f64]]
    shape_dfuncs: Callable[[Vec[f64]], Mat[f64]]

    def write(self, fout: TextIO, elem: Arr[int, i32], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        for j in self.nodeordering:
            fout.write(f" {elem[j] - 1:d}")
        fout.write("\n")

    def __hash__(self) -> int:
        return hash(self.elem)


class VtkType(VtkElemInterface, enum.Enum):
    LineLinear = (
        "VtkLineLinear",
        None,
        3,
        None,
        (0, 1),
        (0, 1),
        np.array([[0, 0, 0], [1, 0, 0]], dtype=int),
        np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        sf_line_linear,
        dsf_line_linear,
    )
    LineQuadratic = (
        "VtkLineQuadratic",
        None,
        21,
        None,
        (0, 1, 2),
        (0, 1, 2),
        np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0]], dtype=int),
        np.array([[0, 0, 0], [1, 0, 0], [1 / 2, 0, 0]], dtype=float),
        sf_line_quadratic,
        dsf_line_quadratic,
    )
    TriangleLinear = (
        "VtkTriangleLinear",
        "VtkLineLinear",
        5,
        3,
        (0, 1, 2),
        (0, 1, 2),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=int),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        sf_triangle_linear,
        dsf_triangle_linear,
    )
    TriangleQuadratic = (
        "VtkTriangleQuadratic",
        "VtkLineQuadratic",
        22,
        21,
        (0, 1, 2, 3, 5, 4),
        (0, 1, 2, 3, 5, 4),
        np.array([[0,0,0], [2,0,0], [0,2,0], [1,0,0], [0,1,0], [1,1,0]], dtype=int),  # fmt: skip
        np.array([[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1/2,0], [1/2,1/2,0]], dtype=float),  # fmt: skip
        sf_triangle_quadratic,
        dsf_triangle_quadratic,
    )
    QuadrilateralLinear = (
        "VtkQuadrilateralLinear",
        "VtkLineLinear",
        9,
        3,
        (0, 1, 3, 2),
        (0, 1, 3, 2),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=int),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        sf_quadrilateral_linear,
        dsf_quadrilateral_linear,
    )
    QuadrilateralQuadratic = (
        "VtkQuadrilateralQuadratic",
        "VtkLineQuadratic",
        28,
        21,
        (0, 1, 3, 2, 4, 7, 8, 5, 6),
        (0, 1, 3, 2, 4, 7, 8, 5, 6),
        np.array([[0,0,0],[2,0,0],[0,2,0],[2,2,0],[1,0,0],[0,1,0],[1,1,0],[2,1,0],[1,2,0]], dtype=float),  # fmt: skip
        np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0.5,0,0],[0,0.5,0],[0.5,0.5,0],[1,0.5,0],[0.5,1,0]], dtype=float),  # fmt: skip
        sf_quadrilateral_quadratic,
        dsf_quadrilateral_quadratic,
    )
    TetrahedronLinear = (
        "VtkTetrahedronLinear",
        "VtkTriangleLinear",
        10,
        5,
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        np.array([0], dtype=int),  # fmt: skip
        np.array([0], dtype=float),  # fmt: skip
        sf_tetrahedron_linear,
        dsf_tetrahedron_linear,
    )
    TetrahedronQuadratic = (
        "VtkTetrahedronQuadratic",
        "VtkTriangleQuadratic",
        24,
        22,
        (0, 1, 2, 3, 4, 6, 5, 7, 8, 9),
        (0, 1, 2, 3, 4, 6, 5, 7, 8, 9),
        np.array([0], dtype=int),  # fmt: skip
        np.array([0], dtype=float),  # fmt: skip
        sf_tetrahedron_quadratic,
        dsf_tetrahedron_quadratic,
    )
    HexahedronLinear = (
        "VtkHexahedronLinear",
        "VtkQuadrilateralLinear",
        12,
        9,
        (0, 1, 5, 4, 2, 3, 7, 6),
        (0, 1, 5, 4, 2, 3, 7, 6),
        np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=int),  # fmt: skip
        np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=float),  # fmt: skip
        sf_hexahedron_linear,
        dsf_hexahedron_linear,
    )
    HexahedronQuadratic = (
        "VtkHexahedronQuadratic",
        "VtkQuadrilateralQuadratic",
        29,
        28,
        (0,1,5,4,2,3,7,6,8,15,22,13,12,21,26,19,9,11,25,23,16,18,10,24,14,20,17),  # fmt: skip
        (0,1,5,4,2,3,7,6,8,15,22,13,12,21,26,19,9,11,25,23,16,18,10,24,14,20,17),  # fmt: skip
        np.array(
            [
                [0,0,0],[2,0,0],[0,2,0],[2,2,0],[0,0,2],[2,0,2],[0,2,2],[2,2,2],
                [1,0,0],[0,1,0],[1,1,0],[2,1,0],[1,2,0],
                [0,0,1],[1,0,1],[2,0,1],[0,1,1],[1,1,1],[2,1,1],[0,2,1],[1,2,1],[2,2,1],
                [1,0,2],[0,1,2],[1,1,2],[2,1,2],[1,2,2],
            ], dtype=int
        ),  # fmt: skip
        0.5*np.array(
            [
                [0,0,0],[2,0,0],[0,2,0],[2,2,0],[0,0,2],[2,0,2],[0,2,2],[2,2,2],
                [1,0,0],[0,1,0],[1,1,0],[2,1,0],[1,2,0],
                [0,0,1],[1,0,1],[2,0,1],[0,1,1],[1,1,1],[2,1,1],[0,2,1],[1,2,1],[2,2,1],
                [1,0,2],[0,1,2],[1,1,2],[2,1,2],[1,2,2],
            ], dtype=float
        ),  # fmt: skip
        sf_hexahedron_quadratic,
        dsf_hexahedron_quadratic,
    )

    # def __new__(
    #     cls,
    #     elem: VTK_ELEM_TYPE,
    #     vtkelementid: int,
    #     vtksurfaceid: int | None,
    #     nodeordering: tuple[int, ...],
    #     connectivity: tuple[int, ...],
    #     shape_funcs: Callable[[Vec[f64]], Vec[f64]],
    #     shape_dfuncs: Callable[[Vec[f64]], Mat[f64]],
    # ):
    #     return super(VtkType, cls).__new__(VtkElemInterface)


VTK_ELEM: Mapping[VTK_ELEM_TYPE | None, VtkType | None] = {
    None: None,
    "VtkLineLinear": VtkType.LineLinear,
    "VtkLineQuadratic": VtkType.LineQuadratic,
    "VtkTriangleLinear": VtkType.TriangleLinear,
    "VtkTriangleQuadratic": VtkType.TriangleQuadratic,
    "VtkQuadrilateralLinear": VtkType.QuadrilateralLinear,
    "VtkQuadrilateralQuadratic": VtkType.QuadrilateralQuadratic,
    "VtkTetrahedronLinear": VtkType.TetrahedronLinear,
    "VtkTetrahedronQuadratic": VtkType.TetrahedronQuadratic,
    "VtkHexahedronLinear": VtkType.HexahedronLinear,
    "VtkHexahedronQuadratic": VtkType.HexahedronQuadratic,
}


def guess_elem_type_from_dim(edim: int, bdim: int | None) -> tuple[VtkType, VtkType]:
    match [edim, bdim]:
        case [3, 2 | None]:
            return VtkType.TriangleLinear, VtkType.LineLinear
        case [6, 3 | None]:
            return VtkType.TriangleQuadratic, VtkType.LineQuadratic
        case [4, 2]:
            return VtkType.QuadrilateralLinear, VtkType.LineLinear
        case [9, 3 | None]:
            return VtkType.QuadrilateralQuadratic, VtkType.LineQuadratic
        case [4, 3]:
            return VtkType.TetrahedronLinear, VtkType.TriangleLinear
        case [10, 6 | None]:
            return VtkType.TetrahedronQuadratic, VtkType.TriangleQuadratic
        case [8, 4 | None]:
            return VtkType.HexahedronLinear, VtkType.QuadrilateralLinear
        case [27, 9 | None]:
            return VtkType.HexahedronQuadratic, VtkType.QuadrilateralQuadratic
        case [4, None]:
            raise ValueError(
                "Cannot detect Bilinear quadrilateral/Trilinear tetrahedron, need boundary dim"
            )
        case _:
            raise ValueError(f"Cannot determine element type from {edim} and {bdim}")
