import enum
from typing import Literal

SolverMatrixOptions = Literal[
    "SolverDependents",
    "SolverMatrixCalculation",
    "matrix_built_by_petsc",
    "ScaleResidualNorms",
    "LeftHandScaling",
    "RightHandScaling",
]

SolverMatrixCalculationOptions = Literal[
    "EVALUATE_EVERY_ITERATION",
    "EVALUATE_EVERY_TIMESTEP",
    "EVALUATE_EVERY_BUILD",
    "EVALUATE_ONCE",
    "EVALUATE_NEVER",
    "EVALUATE_EVERY_TIMEGRID",
    "EVALUATE_ONCE_PER_TIMEGRID",
]


class SolverMatrixCalculationEnum(enum.StrEnum):
    EVALUATE_EVERY_ITERATION = "EVALUATE_EVERY_ITERATION"
    EVALUATE_EVERY_TIMESTEP = "EVALUATE_EVERY_TIMESTEP"
    EVALUATE_EVERY_BUILD = "EVALUATE_EVERY_BUILD"
    EVALUATE_ONCE = "EVALUATE_ONCE"
    EVALUATE_NEVER = "EVALUATE_NEVER"
    EVALUATE_EVERY_TIMEGRID = "EVALUATE_EVERY_TIMEGRID"
    EVALUATE_ONCE_PER_TIMEGRID = "EVALUATE_ONCE_PER_TIMEGRID"


MUMPSMatrixOptions = Literal["MemoryBuffer", "SolverStats", "ordering"]


class MUMPSMatrixEnum(enum.StrEnum):
    MEMORY_BUFFER = "MemoryBuffer"
    SOLVER_STATS = "SolverStats"
    ORDERING = "ordering"


MUMPSOrderingOptions = Literal["sequential", "parallel"]

MUMPSSequentialOrderingOptions = Literal["scotch", "pord", "metis"]
MUMPSParallelOrderingOptions = Literal["ptscotch", "parmetis"]
