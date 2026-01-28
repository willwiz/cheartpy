import enum
from typing import Literal

type TolSetting = Literal[
    "L2TOL",
    "L2PERCENT",
    "INFRES",
    "INFUPDATE",
    "INFDEL",
    "INFRELUPDATE",
    "L2RESRELPERCENT",
]


class TolEnum(enum.StrEnum):
    L2TOL = "L2TOL"
    L2PERCENT = "L2PERCENT"
    INFRES = "INFRES"
    INFUPDATE = "INFUPDATE"
    INFDEL = "INFDEL"
    INFRELUPDATE = "INFRELUPDATE"
    L2RESRELPERCENT = "L2RESRELPERCENT"


type IterationSetting = Literal[
    "ITERATION",
    "SUBITERATION",
    "LINESEARCHITER",
    "SUBITERFRACTION",
    "GroupIterations",
]


class IterationEnum(enum.StrEnum):
    ITERATION = "ITERATION"
    SUBITERATION = "SUBITERATION"
    LINESEARCHITER = "LINESEARCHITER"
    SUBITERFRACTION = "SUBITERFRACTION"
    GroupIterations = "GroupIterations"


type MatrixSolverOption = Literal["SOLVER_MUMPS", "SOLVER_PETSC"]


class MatrixSolverEnum(enum.StrEnum):
    SOLVER_MUMPS = "SOLVER_MUMPS"
    SOLVER_PETSC = "SOLVER_PETSC"


# Solver Algorithms


type SolverSubgroupMethod = Literal["seq_fp", "seq_fp_linesearch", "SOLVER_SEQUENTIAL"]


class SolverSubgroupMethodEnum(enum.StrEnum):
    seq_fp = "seq_fp"
    seq_fp_linesearch = "seq_fp_linesearch"
    SOLVER_SEQUENTIAL = "SOLVER_SEQUENTIAL"


# Solver Group Options


type SolverSubgroupOption = Literal[
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


class SolverSubgroupOptEnum(enum.StrEnum):
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
