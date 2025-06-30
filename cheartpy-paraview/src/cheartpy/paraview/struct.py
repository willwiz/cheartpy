from pathlib import Path

__all__ = [
    "CheartTopology",
    "CmdLineArgs",
    "IFormattedName",
    "ProgramArgs",
    "VariableCache",
]
import dataclasses as dc
from collections.abc import Mapping, Sequence
from typing import Final, Literal, TypedDict

import numpy as np
from arraystubs import Arr1, Arr2
from cheartpy.io.indexing.interfaces import SearchMode
from cheartpy.vtk.api import guess_elem_type_from_dim
from cheartpy.vtk.trait import VtkType
from pytools.logging.trait import LogLevel

from .trait import IFormattedName


class APIKwargs(TypedDict, total=False):
    prefix: str | None
    index: tuple[int, int, int] | None
    subindex: tuple[int, int, int] | Literal["auto", "none"]
    vars: Sequence[str]
    input_dir: str
    output_dir: str
    mesh: str | tuple[str, str, str]
    space: str | None
    time_series: str | None
    binary: bool
    compression: bool
    progress_bar: bool
    cores: int
    cmd: Literal["index", "find"]
    log: LogLevel


@dc.dataclass(slots=True)
class CmdLineArgs:
    mesh: Final[str | tuple[str, str, str]]
    var: Final[Sequence[str]]
    space: Final[str | None] = None
    prefix: Final[str | None] = None
    input_dir: Final[str] = ""
    output_dir: Final[str] = ""
    time_series: Final[str | None] = None
    progress_bar: Final[bool] = True
    log: Final[LogLevel] = LogLevel.INFO
    binary: Final[bool] = False
    compression: Final[bool] = True
    cores: Final[int] = 1
    index: tuple[int, int, int] | SearchMode = SearchMode.none
    subindex: tuple[int, int, int] | SearchMode = SearchMode.none


@dc.dataclass(slots=True, frozen=True)
class ProgramArgs:
    prefix: Final[str]
    input_folder: Final[Path]
    output_folder: Final[Path]
    time_series: Final[str | None]
    progress_bar: Final[bool]
    binary: Final[bool]
    compression: Final[bool]
    cores: Final[int]
    tfile: Final[Path]
    bfile: Final[Path | None]
    space: Final[IFormattedName]
    disp: Final[IFormattedName | None]
    var: Final[Mapping[str, IFormattedName]]


class CheartTopology[I: np.integer]:
    __slots__ = ["_ft", "nc", "ne", "vtkelementtype", "vtksurfacetype"]

    _ft: Arr2[I]
    ne: int
    nc: int
    vtkelementtype: VtkType
    vtksurfacetype: VtkType | None

    def __init__(self, tfile: Path | str, bfile: Path | None, *, dtype: type[I] = np.intc) -> None:
        ################################################################################################
        # read topology and get number of elements, number of nodes per elements
        self._ft = np.loadtxt(tfile, skiprows=1, dtype=dtype) - 1
        if self._ft.ndim == 1:
            self._ft = self._ft[:, np.newaxis]
        self.ne = self._ft.shape[0]
        self.nc = self._ft.shape[1]
        # guess the VTK element type
        # bilinear triangle
        match bfile:
            case Path():
                with Path(bfile).open("r") as f:
                    _ = next(f)  # skip header
                    bdim = len(next(f).strip().split()) - 2
            case None:
                bdim = None
        vtk = guess_elem_type_from_dim(self.nc, bdim)
        self.vtkelementtype, self.vtksurfacetype = vtk.body, vtk.surf

    def __setitem__(self, index: int, data: Arr1[I]) -> None:
        self._ft[index] = data

    def __getitem__(self, index: int) -> int | Arr1[I]:
        return self._ft[index]

    def get_data(self) -> Arr2[I]:
        return self._ft


@dc.dataclass(slots=True)
class VariableCache[F: np.floating, I: np.integer]:
    top: Final[CheartTopology[I]]
    t: str | int
    space_i: Path
    disp_i: Path | None
    space: Arr2[F]
    disp: Arr2[F]
    x: Arr2[F]
    var_i: dict[str, Path] = dc.field(default_factory=dict[str, Path])
    var: dict[str, Arr2[F]] = dc.field(default_factory=dict[str, "Arr2[F]"])


@dc.dataclass(slots=True)
class InputArguments:
    space: str | Arr2[np.float64]
    disp: str | None
    var: dict[str, str]
    prefix: str
