import enum
from collections.abc import Mapping
from typing import Literal, NamedTuple

type NodeOrder = Mapping[int, tuple[int, int, int]]
type VtkElemShape = Literal[
    "Vertex", "Line", "Triangle", "Quadrilateral", "Tetrahedron", "Hexahedron"
]

type ElemType = Literal["Cheart", "Vtk", "Abaqus", "Gmsh"]
type ElemEnum = CheartEnum | VtkEnum | AbaqusEnum | GmshEnum


class _VtkElem(NamedTuple):
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


class VtkEnum(_VtkElem, enum.Enum):
    VERTEX = ("VTK_VERTEX", "Vertex", 0, 1)
    LINE1 = ("VTK_LINE", "Line", 1, 3)
    TRIANGLE1 = ("VTK_TRIANGLE", "Triangle", 1, 5)
    QUADRILATERAL1 = ("VTK_QUAD", "Quadrilateral", 1, 9)
    TETRAHEDRON1 = ("VTK_TETRA", "Tetrahedron", 1, 10)
    HEXAHEDRON1 = ("VTK_HEXAHEDRON", "Hexahedron", 1, 12)
    LINE2 = ("VTK_QUADRATIC_EDGE", "Line", 2, 21)
    TRIANGLE2 = ("VTK_QUADRATIC_TRIANGLE", "Triangle", 2, 22)
    QUADRILATERAL2 = ("VTK_BIQUADRATIC_QUAD", "Quadrilateral", 2, 28)
    TETRAHEDRON2 = ("VTK_QUADRATIC_TETRA", "Tetrahedron", 2, 24)
    HEXAHEDRON2 = ("VTK_QUADRATIC_HEXAHEDRON", "Hexahedron", 2, 29)


class CheartEnum(enum.Enum):
    VERTEX = enum.auto()
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
    VERTEX = "NODE"  # 1-node point element
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
    HEXAHEDRON2 = "C3D27"  # 20-node quadratic 3D hexahedral element


class GmshEnum(enum.IntEnum):
    VERTEX = 15
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
