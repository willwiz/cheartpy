import enum
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence
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


class SolverKwargs(TypedDict, total=False):
    cores: int
    log: bool
    dump_matrix: bool
    verbose: VerbosityEnum
    logger: ILogger


class PrepArgs(TypedDict, total=True):
    pfile: Sequence[Path]


class PrepKwargs(TypedDict, total=False):
    quiet: bool
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
