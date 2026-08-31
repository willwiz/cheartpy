import enum
from collections.abc import Mapping
from typing import Literal, NamedTuple

type NodeOrder = Mapping[int, tuple[int, ...]]
type VtkElemShape = Literal["Line", "Triangle", "Quadrilateral", "Tetrahedron", "Hexahedron"]


class _VtkElem(NamedTuple):
    name: str
    enum: str
    shape: VtkElemShape
    order: int
    idx: int


type VtkElemType = Literal[
    "VtkConstLine",
    "VtkConstTriangle",
    "VtkConstQuadrilateral",
    "VtkConstTetrahedron",
    "VtkConstHexahedron",
    "VtkLinearLine",
    "VtkLinearTriangle",
    "VtkLinearQuadrilateral",
    "VtkLinearTetrahedron",
    "VtkLinearHexahedron",
    "VtkQuadraticLine",
    "VtkQuadraticTriangle",
    "VtkQuadraticQuadrilateral",
    "VtkQuadraticTetrahedron",
    "VtkQuadraticHexahedron",
]


class VtkEnum(enum.Enum):
    LINE1 = _VtkElem("vtkLine", "VTK_LINE", "Line", 1, 3)
    TRIANGLE1 = _VtkElem("vtkTriangle", "VTK_TRIANGLE", "Triangle", 1, 5)
    QUADRILATERAL1 = _VtkElem("vtkQuad", "VTK_QUAD", "Quadrilateral", 1, 9)
    TETRAHEDRON1 = _VtkElem("vtkTetra", "VTK_TETRA", "Tetrahedron", 1, 10)
    HEXAHEDRON1 = _VtkElem("vtkHexahedron", "VTK_HEXAHEDRON", "Hexahedron", 1, 12)
    LINE2 = _VtkElem("vtkQuadraticEdge", "VTK_QUADRATIC_EDGE", "Line", 2, 21)
    TRIANGLE2 = _VtkElem("VtkQuadraticTriangle", "VTK_QUADRATIC_TRIANGLE", "Triangle", 2, 22)
    QUADRILATERAL2 = _VtkElem("VtkBiquadraticQuad", "VTK_BIQUADRATIC_QUAD", "Quadrilateral", 2, 28)
    TETRAHEDRON2 = _VtkElem("VtkQuadraticTetra", "VTK_QUADRATIC_TETRA", "Tetrahedron", 2, 24)
    HEXAHEDRON2 = _VtkElem(
        "VtkQuadraticHexahedron", "VTK_QUADRATIC_HEXAHEDRON", "Hexahedron", 2, 29
    )


class CheartEnum(enum.Enum):
    LINE0 = enum.auto()
    TRIANGLE0 = enum.auto()
    QUADRILATERAL0 = enum.auto()
    TETRAHEDRON0 = enum.auto()
    HEXAHEDRON0 = enum.auto()
    LINE1 = enum.auto()
    TRIANGLE1 = enum.auto()
    QUADRILATERAL1 = enum.auto()
    TETRAHEDRON1 = enum.auto()
    HEXAHEDRON1 = enum.auto()
    LINE2 = enum.auto()
    TRIANGLE2 = enum.auto()
    QUADRILATERAL2 = enum.auto()
    TETRAHEDRON2 = enum.auto()
    HEXAHEDRON2 = enum.auto()


class AbaqusEnum(enum.StrEnum):
    S3R = "S3R"  # 3-node linear shell element with reduced integration
    CPEG6 = "CPEG6"  # 6-node quadratic plane strain element
    LINE1 = "T3D2"  # 2-node linear 3D truss element
    LINE2 = "T3D3"  # 3-node quadratic 3D truss element
    TRIANGLE1 = "CPS3"  # 3-node linear triangle plane stress element
    TRIANGLE2 = "CPS6"  # 6-node quadratic triangle plane stress element
    QUADRILATERAL1 = "CPS4"  # 4-node linear plane stress element
    QUADRILATERAL2 = "M3D9"  # 8-node quadratic plane stress element
    TETRAHEDRON1 = "C3D4"  # 4-node linear 3D tetrahedral element
    TETRAHEDRON2 = "C3D10"  # 10-node quadratic 3D tetrahedral element
    HEXAHEDRON1 = "C3D8"  # 8-node linear 3D hexahedral element
    HEXAHEDRON2 = "C3D20"  # 20-node quadratic 3D hexahedral element


class GmshEnum(enum.IntEnum):
    LINE1 = 1
    TRIANGLE1 = 2
    QUADRILATERAL1 = 3
    TETRAHEDRON1 = 4
    HEXAHEDRON1 = 5
    LINE2 = 8
    TRIANGLE2 = 9
    QUADRILATERAL2 = 10
    TETRAHEDRON2 = 11
    HEXAHEDRON2 = 12
