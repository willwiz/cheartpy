import enum
from typing import Literal

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


MATRIX_SOLVER_OPTIONS = Literal["SOLVER_MUMPS", "SOLVER_PETSC"]


class MatrixSolverOptions(enum.StrEnum):
    MUMPS = "SOLVER_MUMPS"
    PETSC = "SOLVER_PETSC"


# Solver Algorithms


SOLVER_SUBGROUP_ALGORITHM = Literal["seq_fp_linesearch", "seq_fp", "SOLVER_SEQUENTIAL"]


class SolverSubgroupAlgorithm(enum.StrEnum):
    seq_fp = "seq_fp"
    seq_fp_linesearch = "seq_fp_linesearch"
    SOLVER_SEQUENTIAL = "SOLVER_SEQUENTIAL"


# Solver Group Options


SOLVER_SUBGROUP_OPTIONS = Literal[
    "AddVariables",
    "export_initial_condition",
    "L2TOL",
    "L2PERCENT",
    "INFRES",
    "INFUPDATE",
    "INFDEL",
    "ITERATION",
    "SUBITERATION",
    "LINESEARCHITER",
    "SUBITERFRACTION",
]


class SolverSubGroupOptions(enum.StrEnum):
    AddVariables = "AddVariables"
    export_initial_condition = "export_initial_condition"
    L2TOL = "L2TOL"
    L2PERCENT = "L2PERCENT"
    INFRES = "INFRES"
    INFUPDATE = "INFUPDATE"
    INFDEL = "INFDEL"
    ITERATION = "ITERATION"
    SUBITERATION = "SUBITERATION"
    LINESEARCHITER = "LINESEARCHITER"
    SUBITERFRACTION = "SUBITERFRACTION"
    INFRELUPDATE = "INFRELUPDATE"
    L2RESRELPERCENT = "L2RESRELPERCENT"
