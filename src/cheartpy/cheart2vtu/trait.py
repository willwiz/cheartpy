from __future__ import annotations

from pathlib import Path

__all__ = [
    "CheartTopology",
    "CmdLineArgs",
    "IFormattedName",
    "ProgramArgs",
    "ProgramMode",
    "VariableCache",
]
import abc
import dataclasses as dc
import enum
from collections.abc import Sequence
from typing import Final

import numpy as np
from arraystubs import Arr1, Arr2
from pytools.logging.trait import LogLevel

from cheartpy.io.indexing.interfaces import SearchMode
from cheartpy.vtk.api import guess_elem_type_from_dim
from cheartpy.vtk.trait import VtkType


class ProgramMode(enum.StrEnum):
    none = "none"
    search = "search"
    searchsubindex = "searchsubindex"
    range = "range"
    subindex = "subindex"
    subauto = "subauto"


class IFormattedName(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, i: str | int) -> Path: ...


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
    input_folder: Final[str]
    output_folder: Final[str]
    time_series: Final[str | None]
    progress_bar: Final[bool]
    binary: Final[bool]
    compression: Final[bool]
    cores: Final[int]
    tfile: Final[str]
    bfile: Final[str | None]
    space: Final[IFormattedName]
    disp: Final[IFormattedName | None]
    var: Final[dict[str, IFormattedName]]


class CheartTopology[I: np.integer]:
    __slots__ = ["_ft", "nc", "ne", "vtkelementtype", "vtksurfacetype"]

    _ft: Arr2[I]
    ne: int
    nc: int
    vtkelementtype: VtkType
    vtksurfacetype: VtkType | None

    def __init__(self, tfile: str, bfile: str | None) -> None:
        ################################################################################################
        # read topology and get number of elements, number of nodes per elements
        self._ft = np.loadtxt(tfile, skiprows=1, dtype=int)
        if self._ft.ndim == 1:
            self._ft = np.array([self._ft])
        self.ne = self._ft.shape[0]
        self.nc = self._ft.shape[1]
        # guess the VTK element type
        # bilinear triangle
        vtk = guess_elem_type_from_dim(self.nc, bfile)
        self.vtkelementtype, self.vtksurfacetype = vtk.elem, vtk.surf

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
    space_i: str
    disp_i: str | None
    space: Arr2[F]
    disp: Arr2[F]
    x: Arr2[F]
    var_i: dict[str, Path] = dc.field(default_factory=dict[str, Path])
    var: dict[str, Arr2[F]] = dc.field(default_factory=dict[str, Arr2[F]])


@dc.dataclass(slots=True)
class InputArguments:
    space: str | Arr2[np.float64]
    disp: str | None
    var: dict[str, str]
    prefix: str
