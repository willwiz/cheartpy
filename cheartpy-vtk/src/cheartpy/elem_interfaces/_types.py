import enum
from typing import Literal, NamedTuple

type VtkElemShape = Literal["Line", "Triangle", "Quadrilateral", "Tetrahedron", "Hexahedron"]


class _VtkElem(NamedTuple):
    name: str
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
    VtkConstLine = _VtkElem("VtkConstLine", "Line", 0, -1)
    VtkConstTriangle = _VtkElem("VtkConstTriangle", "Triangle", 0, -2)
    VtkConstQuadrilateral = _VtkElem("VtkConstQuadrilateral", "Quadrilateral", 0, -3)
    VtkConstTetrahedron = _VtkElem("VtkConstTetrahedron", "Tetrahedron", 0, -4)
    VtkConstHexahedron = _VtkElem("VtkConstHexahedron", "Hexahedron", 0, -5)
    VtkLinearLine = _VtkElem("VtkLinearLine", "Line", 1, 3)
    VtkLinearTriangle = _VtkElem("VtkLinearTriangle", "Triangle", 1, 5)
    VtkLinearQuadrilateral = _VtkElem("VtkLinearQuadrilateral", "Quadrilateral", 1, 9)
    VtkLinearTetrahedron = _VtkElem("VtkLinearTetrahedron", "Tetrahedron", 1, 10)
    VtkLinearHexahedron = _VtkElem("VtkLinearHexahedron", "Hexahedron", 1, 12)
    VtkQuadraticLine = _VtkElem("VtkQuadraticLine", "Line", 2, 21)
    VtkQuadraticTriangle = _VtkElem("VtkQuadraticTriangle", "Triangle", 2, 22)
    VtkQuadraticQuadrilateral = _VtkElem("VtkQuadraticQuadrilateral", "Quadrilateral", 2, 28)
    VtkQuadraticTetrahedron = _VtkElem("VtkQuadraticTetrahedron", "Tetrahedron", 2, 24)
    VtkQuadraticHexahedron = _VtkElem("VtkQuadraticHexahedron", "Hexahedron", 2, 29)


type CheartElemType = Literal[
    "LINE0",
    "TRIANGLE0",
    "QUADRILATERAL0",
    "TETRAHEDRON0",
    "HEXAHEDRON0",
    "LINE1",
    "TRIANGLE1",
    "QUADRILATERAL1",
    "TETRAHEDRON1",
    "HEXAHEDRON1",
    "LINE2",
    "TRIANGLE2",
    "QUADRILATERAL2",
    "TETRAHEDRON2",
    "HEXAHEDRON2",
]


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
    T3D2 = "T3D2"
    T3D3 = "T3D3"
    CPS3 = "CPS3"
    CPS4 = "CPS4"
    CPS6 = "CPS6"
    C3D4 = "C3D4"
    S3R = "S3R"
    C3D10 = "C3D10"
    CPEG6 = "CPEG6"


type AbaqusElemType = Literal[
    "T3D2",
    "T3D3",
    "CPS3",
    "CPS4",
    "CPS6",
    "C3D4",
    "S3R",
    "C3D10",
    "CPEG6",
]
