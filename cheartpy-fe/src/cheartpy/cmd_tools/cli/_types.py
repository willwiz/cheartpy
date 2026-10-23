import enum
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from pytools.logging import ILogger

type Verbosity = Literal["NONE", "DEFAULT", "PEDANTIC", "QUIET"]


class VerbosityEnum(enum.Enum):
    NONE = 0
    DEFAULT = 1
    PEDANTIC = 2
    QUIET = 3


class SolverArgs(TypedDict, total=True):
    pfile: Sequence[Path]
    parallel: int


class SolverKwargs(TypedDict, total=False):
    #: The number of mpi processes to use. If 1 if not given.
    cores: int
    #: Log the cheart solver output to a file with the same name as the pfile but with a .log
    #: extension. If False, the output will be printed to stdout.
    log: bool
    #: Dump the matrix to a file. Name and location is determined by Cheart.
    dump_matrix: bool
    #: Dump the residual to a file. Name and location is determined by Cheart.
    dump_residual: bool
    #: Dump the intermediate results to a file. Name and location is determined by Cheart.
    dump_intermediate: bool
    #: Clear the output directory before running.
    clear: bool
    #: Macro values to be used by the pfile
    macros: Mapping[str, str]
    #: The MPI implementation to use. If None, no optimization for mpi will be used.
    opt: Literal["mpich", "openmpi", "none"] | None
    #: The verbosity level for Cheart
    verbosity: Verbosity
    #: Use a logger to intercept the output of Cheart.
    logger: ILogger


class PrepArgs(TypedDict, total=True):
    pfile: Sequence[Path]


class PrepKwargs(TypedDict, total=False):
    verbosity: Verbosity
    log: bool
    logger: ILogger


class CheartErrorCode(enum.Enum):
    SUCCESS = 0
    UNSPECIFIED = 1
    PFILE = 2
    CONVERGENCE_FAILED = 3
    MUMPS = 4
    DEV = 10
    UNKNOWN = 255
