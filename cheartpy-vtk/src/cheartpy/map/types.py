import enum


class VtkEnum(enum.Enum):
    VtkLinearLine = "VtkLinearLine"
    VtkLinearTriangle = "VtkLinearTriangle"
    VtkLinearQuadrilateral = "VtkLinearQuadrilateral"
    VtkLinearTetrahedron = "VtkLinearTetrahedron"
    VtkLinearHexahedron = "VtkLinearHexahedron"
    VtkQuadraticLine = "VtkQuadraticLine"
    VtkQuadraticTriangle = "VtkQuadraticTriangle"
    VtkQuadraticQuadrilateral = "VtkQuadraticQuadrilateral"
    VtkQuadraticTetrahedron = "VtkQuadraticTetrahedron"
    VtkQuadraticHexahedron = "VtkQuadraticHexahedron"


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
    CPS4_3D = "CPS4_3D"
    C3D4 = "C3D4"
    S3R = "S3R"
    TetQuad3D = "TetQuad3D"
    Tet3D = "Tet3D"
