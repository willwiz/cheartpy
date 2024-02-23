import enum
from typing import Literal


class TolSettings(enum.StrEnum):
    L2TOL = "L2TOL"
    L2PERCENT = "L2PERCENT"
    INFRES = "INFRES"
    INFUPDATE = "INFUPDATE"
    INFDEL = "INFDEL"
    INFRELUPDATE = "INFRELUPDATE"
    L2RESRELPERCENT = "L2RESRELPERCENT"


class IterationSettings(enum.StrEnum):
    ITERATION = "ITERATION"
    SUBITERATION = "SUBITERATION"
    LINESEARCHITER = "LINESEARCHITER"
    SUBITERFRACTION = "SUBITERFRACTION"
    GroupIterations = "GroupIterations"


class InterfaceTypes(enum.StrEnum):
    OneToOne = "OneToOne"
    ManyToOne = "ManyToOne"


VARIABLE_EXPORT_FORMAT = Literal["TXT", "BINARY", "MMAP"]


class VariableExportFormat(enum.StrEnum):
    TXT = "TXT"
    BINARY = "ReadBinary"
    MMAP = "ReadMMap"


class SolverOptions(enum.StrEnum):
    MUMPS = "SOLVER_MUMPS"


class SolidProblems(enum.StrEnum):
    TRANSIENT = "transient_elasticity"
    QUASI_STATIC = "quasi_static_elasticity"


class L2ProjectionProblems(enum.StrEnum):
    SOLID_PROJECTION = "l2solidprojection_problem"


class SolidProjectionProbOptions(enum.StrEnum):
    MASTER_OVERRIDE = "Solid-Master-Override"


CHEART_ELEMENT_TYPE = Literal[
    "POINT_ELEMENT",
    "POINT_ELEMENT",
    "ONED_ELEMENT",
    "ONED_ELEMENT",
    "QUADRILATERAL_ELEMENT",
    "QUADRILATERAL_ELEMENT",
    "TRIANGLE_ELEMENT",
    "TRIANGLE_ELEMENT",
    "HEXAHEDRAL_ELEMENT",
    "HEXAHEDRAL_ELEMENT",
    "TETRAHEDRAL_ELEMENT",
    "TETRAHEDRAL_ELEMENT",
]


class CheartElementType(enum.StrEnum):
    POINT_ELEMENT = "POINT_ELEMENT"
    point = "POINT_ELEMENT"
    ONED_ELEMENT = "ONED_ELEMENT"
    line = "ONED_ELEMENT"
    QUADRILATERAL_ELEMENT = "QUADRILATERAL_ELEMENT"
    quad = "QUADRILATERAL_ELEMENT"
    TRIANGLE_ELEMENT = "TRIANGLE_ELEMENT"
    tri = "TRIANGLE_ELEMENT"
    HEXAHEDRAL_ELEMENT = "HEXAHEDRAL_ELEMENT"
    hex = "HEXAHEDRAL_ELEMENT"
    TETRAHEDRAL_ELEMENT = "TETRAHEDRAL_ELEMENT"
    tet = "TETRAHEDRAL_ELEMENT"


class CheartBasisType(enum.StrEnum):
    NODAL_LAGRANGE = "NODAL_LAGRANGE"
    NL = "NL"
    MODAL_BASIS = "MODAL_BASIS"
    PNODAL_BASIS = "PNODAL_BASIS"
    MINI_BASIS = "MINI_BASIS"
    NURBS_BASIS = "NURBS_BASIS"
    SPECTRAL_BASIS = "SPECTRAL_BASIS"


class CheartQuadratureType(enum.StrEnum):
    GAUSS_LEGENDRE = "GAUSS_LEGENDRE"
    GL = "GAUSS_LEGENDRE"
    KEAST_LYNESS = "KEAST_LYNESS"
    KL = "KEAST_LYNESS"
