import enum


class TolSettings(enum.StrEnum):
    L2TOL = "L2TOL"
    L2PERCENT = enum.auto()
    INFRES = enum.auto()
    INFUPDATE = enum.auto()
    INFDEL = enum.auto()
    INFRELUPDATE = enum.auto()
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
