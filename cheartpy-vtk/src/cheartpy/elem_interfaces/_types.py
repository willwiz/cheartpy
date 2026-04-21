import enum
from typing import Literal, NamedTuple


class _VtkElem(NamedTuple):
    name: str
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
    VtkConstLine = _VtkElem("VtkConstLine", -1)
    VtkConstTriangle = _VtkElem("VtkConstTriangle", -2)
    VtkConstQuadrilateral = _VtkElem("VtkConstQuadrilateral", -3)
    VtkConstTetrahedron = _VtkElem("VtkConstTetrahedron", -4)
    VtkConstHexahedron = _VtkElem("VtkConstHexahedron", -5)
    VtkLinearLine = _VtkElem("VtkLinearLine", 3)
    VtkLinearTriangle = _VtkElem("VtkLinearTriangle", 5)
    VtkLinearQuadrilateral = _VtkElem("VtkLinearQuadrilateral", 9)
    VtkLinearTetrahedron = _VtkElem("VtkLinearTetrahedron", 10)
    VtkLinearHexahedron = _VtkElem("VtkLinearHexahedron", 12)
    VtkQuadraticLine = _VtkElem("VtkQuadraticLine", 21)
    VtkQuadraticTriangle = _VtkElem("VtkQuadraticTriangle", 22)
    VtkQuadraticQuadrilateral = _VtkElem("VtkQuadraticQuadrilateral", 28)
    VtkQuadraticTetrahedron = _VtkElem("VtkQuadraticTetrahedron", 24)
    VtkQuadraticHexahedron = _VtkElem("VtkQuadraticHexahedron", 29)


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
