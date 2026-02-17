import enum
from typing import Literal

CheartBasisType = Literal[
    "NODAL_LAGRANGE",
    "NL",
    "MODAL_BASIS",
    "PNODAL_BASIS",
    "MINI_BASIS",
    "NURBS_BASIS",
    "SPECTRAL_BASIS",
]


class CheartBasisEnum(enum.StrEnum):
    NODAL_LAGRANGE = "NL"
    NL = "NL"
    MODAL_BASIS = "MODAL_BASIS"
    PNODAL_BASIS = "PNODAL_BASIS"
    MINI_BASIS = "MINI_BASIS"
    NURBS_BASIS = "NURBS_BASIS"
    SPECTRAL_BASIS = "SPECTRAL_BASIS"


CheartQuadratureType = Literal[
    "GAUSS_LEGENDRE",
    "GL",
    "KEAST_LYNESS",
    "KL",
]


class CheartQuadratureEnum(enum.StrEnum):
    GAUSS_LEGENDRE = "GL"
    GL = "GL"
    KEAST_LYNESS = "KL"
    KL = "KL"


CheartElementType = Literal[
    "POINT_ELEMENT",
    "point",
    "ONED_ELEMENT",
    "line",
    "TRIANGLE_ELEMENT",
    "tri",
    "QUADRILATERAL_ELEMENT",
    "quad",
    "TETRAHEDRAL_ELEMENT",
    "tet",
    "HEXAHEDRAL_ELEMENT",
    "hex",
]


class CheartElementEnum(enum.StrEnum):
    POINT_ELEMENT = "POINT_ELEMENT"
    point = "POINT_ELEMENT"
    ONED_ELEMENT = "ONED_ELEMENT"
    line = "ONED_ELEMENT"
    TRIANGLE_ELEMENT = "TRIANGLE_ELEMENT"
    tri = "TRIANGLE_ELEMENT"
    QUADRILATERAL_ELEMENT = "QUADRILATERAL_ELEMENT"
    quad = "QUADRILATERAL_ELEMENT"
    TETRAHEDRAL_ELEMENT = "TETRAHEDRAL_ELEMENT"
    tet = "TETRAHEDRAL_ELEMENT"
    HEXAHEDRAL_ELEMENT = "HEXAHEDRAL_ELEMENT"
    hex = "HEXAHEDRAL_ELEMENT"


CheartTopologySetting = Literal[
    "PartitioningWeight",
    "UseInPartitioning",
    "ReadBinary",
    "ReadMMap",
    "MakeDiscontinuous",
    "SpatialConstant",
    "CreateInBoundary",
    "EmbeddedInTopology",
]


class CheartTopologyEnum(enum.StrEnum):
    PartitioningWeight = "PartitioningWeight"
    UseInPartitioning = "UseInPartitioning"
    ReadBinary = "ReadBinary"
    ReadMMap = "ReadMMap"
    MakeDiscontinuous = "MakeDiscontinuous"
    SpatialConstant = "SpatialConstant"
    CreateInBoundary = "CreateInBoundary"
    EmbeddedInTopology = "EmbeddedInTopology"


CheartTopInterfaceType = Literal[
    "OneToOne",
    "ManyToOne",
]


class CheartTopInterfaceEnum(enum.StrEnum):
    OneToOne = "OneToOne"
    ManyToOne = "ManyToOne"
