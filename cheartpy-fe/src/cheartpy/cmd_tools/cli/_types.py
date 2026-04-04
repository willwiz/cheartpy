import enum
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class Verbosity(enum.Enum):
    NONE = 0
    PEDANTIC = 1
    QUIET = 2


class SolverArgs(TypedDict, total=True):
    pfile: Sequence[Path]


class SolverKwargs(TypedDict, total=False):
    cores: int
    log: bool
    dump_matrix: bool
    verbose: bool
    quiet: bool


class PrepArgs(TypedDict, total=True):
    pfile: Sequence[Path]


class PrepKwargs(TypedDict, total=False):
    quiet: bool
    log: bool
