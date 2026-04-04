import abc
import dataclasses as dc
from typing import TYPE_CHECKING, ClassVar, Final, TypedDict

from pytools.logging import BColors, LogLevel

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


__all__ = ["IVariableGetter", "IVariableList", "VarErrors", "VarStats"]


HEADER = ["mean", "std", "min", "min pos", "max", "max pos", "bias"]
HEADER_LEN = 8 + 10 + len(HEADER) * 11


class VarCompAPIArgs(TypedDict, total=False):
    var_1: str
    var_2: str


class VarCompAPIKwargs(TypedDict, total=False):
    root_1: Path
    root_2: Path
    log_level: LogLevel


@dc.dataclass(slots=True)
class IVariableList:
    name: Final[str]
    idx: list[int] | None


class IVariableGenerator(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[Path]: ...


class IVariableGetter(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[tuple[int | str, Path, Path]]: ...


@dc.dataclass(slots=True)
class VarErrors:
    mean: float
    sem: float
    vmin: float
    vmin_pos: tuple[int, int]
    vmax: float
    vmax_pos: tuple[int, int]
    rmean: float
    tol: ClassVar[float] = 1e-8

    def __repr__(self) -> str:
        return "|".join([hstr(x, self.tol) for x in dc.astuple(self)])


@dc.dataclass(slots=True)
class VarStats:
    avg: float
    err: VarErrors

    def __repr__(self) -> str:
        return f"{self.avg:>9.2E} ::{self.err}"


def hstr(x: float | tuple[int, int], tol: float = 1e-10) -> str:
    match x:
        case tuple():
            return f"{f'{x[0]},{x[1]}':^9}"
        case _:
            if x > tol:
                return f"{BColors.WARN}{x:>9.2E}{BColors.ENDC}"
            return f"{x:>9.2E}"
