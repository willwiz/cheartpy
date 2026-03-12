import enum
from typing import NamedTuple


class _VtkElem(NamedTuple):
    name: str
    idx: int


class VtkEnum(enum.Enum):
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


class CheartEnum(enum.Enum):
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
    C3D4 = "C3D4"
    S3R = "S3R"
    C3D10 = "C3D10"
    CPEG6 = "CPEG6"
