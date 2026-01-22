import abc
import dataclasses as dc
from typing import TYPE_CHECKING, ClassVar, Final

from pytools.logging.trait import BColors

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


__all__ = ["IVariableGetter", "IVariableList", "VarErrors", "VarStats"]


HEADER = ["mean", "std", "min", "min pos", "max", "max pos", "bias"]
HEADER_LEN = 8 + 10 + len(HEADER) * 11


@dc.dataclass(slots=True)
class IVariableList:
    name: Final[str]
    idx: list[int] | None


class IVariableGetter(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[tuple[int | str, Path | None, Path | None]]: ...


@dc.dataclass(slots=True)
class VarErrors:
    mean: float
    sem: float
    vmin: float
    vmin_pos: tuple[int, int]
    vmax: float
    vmax_pos: tuple[int, int]
    rmean: float
    tol: ClassVar[float] = 1e-10

    def __repr__(self) -> str:
        return "|".join([hstr(x, self.tol) for x in dc.astuple(self)])


@dc.dataclass(slots=True)
class VarStats:
    avg: float
    err: VarErrors

    def __repr__(self) -> str:
        return f"{self.avg:>8.1E}||{self.err}"


def hstr(x: float | tuple[int, int], tol: float = 1e-10) -> str:
    match x:
        case tuple():
            return f"{f'{x[0]},{x[1]}':^10}"
        case _:
            if x > tol:
                return f"{BColors.WARN}{x:>10.3E}{BColors.ENDC}"
            return f"{x:>10.3E}"
