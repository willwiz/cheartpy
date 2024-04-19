import enum
from typing import Literal, Final


TOL_SETTINGS = Literal[
    "L2TOL",
    "L2PERCENT",
    "INFRES",
    "INFUPDATE",
    "INFDEL",
    "INFRELUPDATE",
    "L2RESRELPERCENT",
]


class TolSettings(enum.StrEnum):
    L2TOL = "L2TOL"
    L2PERCENT = "L2PERCENT"
    INFRES = "INFRES"
    INFUPDATE = "INFUPDATE"
    INFDEL = "INFDEL"
    INFRELUPDATE = "INFRELUPDATE"
    L2RESRELPERCENT = "L2RESRELPERCENT"


ITERATION_SETTINGS = Literal[
    "ITERATION",
    "SUBITERATION",
    "LINESEARCHITER",
    "SUBITERFRACTION",
    "GroupIterations",
]


class IterationSettings(enum.StrEnum):
    ITERATION = "ITERATION"
    SUBITERATION = "SUBITERATION"
    LINESEARCHITER = "LINESEARCHITER"
    SUBITERFRACTION = "SUBITERFRACTION"
    GroupIterations = "GroupIterations"


INTERFACE_TYPE = Literal[
    "OneToOne",
    "ManyToOne",
]


class InterfaceTypes(enum.StrEnum):
    OneToOne = "OneToOne"
    ManyToOne = "ManyToOne"


VARIABLE_EXPORT_FORMAT = Literal["TXT", "BINARY", "MMAP"]


class VariableExportFormat(enum.StrEnum):
    TXT = "TXT"
    BINARY = "ReadBinary"
    MMAP = "ReadMMap"


VARIABLE_UPDATE_SETTING = Literal[
    "INIT_EXPR",
    "TEMPORAL_UPDATE_EXPR",
    "TEMPORAL_UPDATE_FILE",
    "TEMPORAL_UPDATE_FILE_LOOP",
]


class VariableUpdateSetting(enum.StrEnum):
    INIT_EXPR = "INIT_EXPR"
    TEMPORAL_UPDATE_EXPR = "TEMPORAL_UPDATE_EXPR"
    TEMPORAL_UPDATE_FILE = "TEMPORAL_UPDATE_FILE"
    TEMPORAL_UPDATE_FILE_LOOP = "TEMPORAL_UPDATE_FILE_LOOP"


class SolverOptions(enum.StrEnum):
    MUMPS = "SOLVER_MUMPS"


SOLID_PROBLEM_TYPE = Literal[
    "TRANSIENT",
    "QUASI_STATIC",
]


class SolidProblemType(enum.StrEnum):
    TRANSIENT = "transient_elasticity"
    QUASI_STATIC = "quasi_static_elasticity"


BOUNDARY_TYPE = Literal[
    "dirichlet",
    "neumann",
    "neumann_ref",
    "neumann_nl",
    "stabilized_neumann",
    "consistent",
]


class BoundaryType(enum.StrEnum):
    dirichlet = "dirichlet"
    neumann = "neumann"
    neumann_ref = "neumann_ref"
    neumann_nl = "neumann_nl"
    stabilized_neumann = "stabilized_neumann"
    consistent = "consistent"


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


CHEART_BASES_TYPE = Literal[
    "NODAL_LAGRANGE",
    "NL",
    "MODAL_BASIS",
    "PNODAL_BASIS",
    "MINI_BASIS",
    "NURBS_BASIS",
    "SPECTRAL_BASIS",
]


class CheartBasisType(enum.StrEnum):
    NODAL_LAGRANGE = "NODAL_LAGRANGE"
    NL = "NL"
    MODAL_BASIS = "MODAL_BASIS"
    PNODAL_BASIS = "PNODAL_BASIS"
    MINI_BASIS = "MINI_BASIS"
    NURBS_BASIS = "NURBS_BASIS"
    SPECTRAL_BASIS = "SPECTRAL_BASIS"


CHEART_QUADRATURE_TYPE = Literal[
    "GAUSS_LEGENDRE",
    "GAUSS_LEGENDRE",
    "KEAST_LYNESS",
    "KEAST_LYNESS",
]


class CheartQuadratureType(enum.StrEnum):
    GAUSS_LEGENDRE = "GAUSS_LEGENDRE"
    GL = "GAUSS_LEGENDRE"
    KEAST_LYNESS = "KEAST_LYNESS"
    KL = "KEAST_LYNESS"


TOPOLOGY_SETTING = Literal[
    "PartitioningWeight",
    "UseInPartitioning",
    "ReadBinary",
    "ReadMMap",
    "MakeDiscontinuous",
    "SpatialConstant",
    "CreateInBoundary",
]


class CheartTopologySetting(enum.StrEnum):
    PartitioningWeight = "PartitioningWeight"
    UseInPartitioning = "UseInPartitioning"
    ReadBinary = "ReadBinary"
    ReadMMap = "ReadMMap"
    MakeDiscontinuous = "MakeDiscontinuous"
    SpatialConstant = "SpatialConstant"
    CreateInBoundary = "CreateInBoundary"


TOPOLOGY_INTERFACE_TYPE = Literal["OneToOne", "ManyToOne"]


class TopologyInterfaceType(enum.StrEnum):
    OneToOne = "OneToOne"
    ManyToOne = "ManyToOne"


MATRIX_SOLVER_TYPES = Literal["SOLVER_MUMPS",]


class MatrixSolverTypes(enum.StrEnum):
    SOLVER_MUMPS = "SOLVER_MUMPS"


# Solver Algorithms


SOLVER_SUBGROUP_ALGORITHM = Literal["seq_fp_linesearch", "seq_fp", "SOLVER_SEQUENTIAL"]


class SolverSubgroupAlgorithm(enum.StrEnum):
    seq_fp_linesearch = "seq_fp_linesearch"
    SOLVER_SEQUENTIAL = "SOLVER_SEQUENTIAL"


class OPTIONS_BASIS:
    pass


class L2ProjectionProblems(enum.StrEnum):
    SOLID_PROJECTION = "l2solidprojection_problem"


class SolidProjectionProbOptions(enum.StrEnum):
    MASTER_OVERRIDE = "Solid-Master-Override"


# Solver Group Options


class OPTIONS_SG:
    AddVariables: Final = "AddVariables"
    export_initial_condition: Final = "export_initial_condition"
    L2TOL: Final = "L2TOL"
    L2PERCENT: Final = "L2PERCENT"
    INFRES: Final = "INFRES"
    INFUPDATE: Final = "INFUPDATE"
    INFDEL: Final = "INFDEL"
    ITERATION: Final = "ITERATION"
    SUBITERATION: Final = "SUBITERATION"
    LINESEARCHITER: Final = "LINESEARCHITER"
    SUBITERFRACTION: Final = "SUBITERFRACTION"
    INFRELUPDATE: Final = "INFRELUPDATE"
    L2RESRELPERCENT: Final = "L2RESRELPERCENT"


# Variable Settings
class OPTIONS_VARIABLE:
    INIT_EXPR: Final = "INIT_EXPR"
    TEMPORAL_UPDATE_EXPR: Final = "TEMPORAL_UPDATE_EXPR"
    TEMPORAL_UPDATE_FILE: Final = "TEMPORAL_UPDATE_FILE"
    ReadBinary: Final = "ReadBinary"
    ReadMMap: Final = "ReadMMap"


# Solvers
class OPTIONS_SOLVER:
    SOLVER_MUMPS: Final = "SOLVER_MUMPS"


# Solid Problems
class OPTIONS_SOLIDPROBLEM:
    transient_elasticity: Final = "transient_elasticity"
    quasi_static_elasticity: Final = "quasi_static_elasticity"


class OPTIONS_L2PROJECTION:
    l2solidprojection_problem: Final = "l2solidprojection_problem"
    Solid_Master_Override: Final = "Solid-Master-Override"


class OPTIONS_PROBLEMS:
    SOLID = OPTIONS_SOLIDPROBLEM()
    L2 = OPTIONS_L2PROJECTION()
    norm_calculation: Final = "norm_calculation"


# Element types
class OPTIONS_ELEMENT:
    HEXAHEDRAL_ELEMENT: Final = "HEXAHEDRAL_ELEMENT"


# Topology Settings


class OPTIONS_TOPOLOGY:
    EmbeddedInTopology: Final = "EmbeddedInTopology"


# Boundary Conditions
class OPTIONS_BC:
    Dirichlet: Final = "Dirichlet"


class OPTIONS_MATLAWS:
    neohookean: Final = "neohookean"
