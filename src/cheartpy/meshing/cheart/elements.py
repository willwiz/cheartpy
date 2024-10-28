import dataclasses as dc
import enum
from typing import Callable, Final, Literal, TextIO
from ...var_types import *
from .shape_functions import *
from .data import CheartMesh

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
    vtkelementid: Final[int]
    vtksurfaceid: Final[int | None]
    nodeordering: Final[tuple[int, ...]]
    connectivity: Final[tuple[int, ...]]
    ref_nodes: Mat[f64]
    shape_funcs: Callable[[Vec[f64]], Vec[f64]]
    shape_dfuncs: Callable[[Vec[f64]], Mat[f64]]

    def write(self, fout: TextIO, elem: Arr[int, i32], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        for j in self.nodeordering:
            fout.write(f" {elem[j] - 1:d}")
        fout.write("\n")


class VtkType(VtkElemInterface, enum.Enum):
    LineLinear = (
        "VtkLineLinear",
        3,
        None,
        (0, 1),
        (0, 1),
        np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        sf_line_linear,
        dsf_line_linear,
    )
    LineQuadratic = (
        "VtkLineQuadratic",
        21,
        None,
        (0, 1, 2),
        (0, 1, 2),
        np.array([[0, 0, 0], [1, 0, 0], [1 / 2, 0, 0]], dtype=float),
        sf_line_quadratic,
        dsf_line_quadratic,
    )
    TriangleLinear = (
        "VtkTriangleLinear",
        5,
        3,
        (0, 1, 2),
        (0, 1, 2),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        sf_triangle_linear,
        dsf_triangle_linear,
    )
    TriangleQuadratic = (
        "VtkTriangleQuadratic",
        22,
        21,
        (0, 1, 2, 3, 5, 4),
        (0, 1, 2, 3, 5, 4),
        np.array([[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1/2,0], [1/2,1/2,0]], dtype=float),  # fmt: skip
        sf_triangle_quadratic,
        dsf_triangle_quadratic,
    )
    QuadrilateralLinear = (
        "VtkQuadrilateralLinear",
        9,
        3,
        (0, 1, 3, 2),
        (0, 1, 3, 2),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        sf_quadrilateral_linear,
        dsf_quadrilateral_linear,
    )
    QuadrilateralQuadratic = (
        "VtkQuadrilateralQuadratic",
        28,
        21,
        (0, 1, 3, 2, 4, 7, 8, 5, 6),
        (0, 1, 3, 2, 4, 7, 8, 5, 6),
        np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[1/2,0,0],[0,1/2,0],[1/2,1/2,0],[1,1/2,0],[1/2,1,0]], dtype=float),  # fmt: skip
        sf_quadrilateral_quadratic,
        dsf_quadrilateral_quadratic,
    )
    TetrahedronLinear = (
        "VtkTetrahedronLinear",
        10,
        5,
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        np.array([0], dtype=float),  # fmt: skip
        sf_tetrahedron_linear,
        dsf_tetrahedron_linear,
    )
    TetrahedronQuadratic = (
        "VtkTetrahedronQuadratic",
        24,
        22,
        (0, 1, 2, 3, 4, 6, 5, 7, 8, 9),
        (0, 1, 2, 3, 4, 6, 5, 7, 8, 9),
        np.array([0], dtype=float),  # fmt: skip
        sf_tetrahedron_quadratic,
        dsf_tetrahedron_quadratic,
    )
    HexahedronLinear = (
        "VtkHexahedronLinear",
        12,
        9,
        (0, 1, 5, 4, 2, 3, 7, 6),
        (0, 1, 5, 4, 2, 3, 7, 6),
        np.array([0], dtype=float),  # fmt: skip
        sf_hexahedron_linear,
        dsf_hexahedron_linear,
    )
    HexahedronQuadratic = (
        "VtkHexahedronQuadratic",
        29,
        28,
        (0,1,5,4,2,3,7,6,8,15,22,13,12,21,26,19,9,11,25,23,16,18,10,24,14,20,17),  # fmt: skip
        (0,1,5,4,2,3,7,6,8,15,22,13,12,21,26,19,9,11,25,23,16,18,10,24,14,20,17),  # fmt: skip
        np.array([0], dtype=float),  # fmt: skip
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


VtkTopologyElement = Literal[
    VtkType.TriangleLinear,
    VtkType.TriangleQuadratic,
    VtkType.QuadrilateralLinear,
    VtkType.QuadrilateralQuadratic,
    VtkType.TetrahedronLinear,
    VtkType.TetrahedronQuadratic,
    VtkType.HexahedronLinear,
    VtkType.HexahedronQuadratic,
]

VtkBoundaryElement = Literal[
    VtkType.LineLinear,
    VtkType.LineQuadratic,
    VtkType.TriangleLinear,
    VtkType.TriangleQuadratic,
    VtkType.QuadrilateralLinear,
    VtkType.QuadrilateralQuadratic,
]


def get_element_type(mesh: CheartMesh) -> tuple[VtkElemInterface, VtkElemInterface]:
    if mesh.bnd is None:
        nbnd = None
    else:
        nbnd = next(iter(mesh.bnd.v.values())).v.shape[1] + 2
    match [mesh.top.v.shape[1], nbnd]:
        case [3, _]:
            return VtkType.TriangleLinear, VtkType.LineLinear
        case [6, _]:
            return VtkType.TriangleQuadratic, VtkType.LineQuadratic
        case [4, 4]:
            return VtkType.QuadrilateralLinear, VtkType.LineLinear
        case [9, _]:
            return VtkType.QuadrilateralQuadratic, VtkType.LineQuadratic
        case [4, 5]:
            return VtkType.TetrahedronLinear, VtkType.TriangleLinear
        case [10, _]:
            return VtkType.TetrahedronQuadratic, VtkType.TriangleQuadratic
        case [8, _]:
            return VtkType.HexahedronLinear, VtkType.QuadrilateralLinear
        case [27, _]:
            return VtkType.HexahedronQuadratic, VtkType.QuadrilateralQuadratic
        case [4, None]:
            raise ValueError(
                "Bilinear quadrilateral / Trilinear tetrahedron detected, need boundary file"
            )
        case [4, _]:
            raise ValueError(
                f"Bilinear quadrilateral / {nbnd}, boundary file is incompatible"
            )
        case _:
            raise ValueError(
                f"Cannot determine element type from {mesh.top.v.shape[1]} and {
                    nbnd}, perhaps not implemented."
            )
