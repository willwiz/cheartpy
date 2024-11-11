__all__ = [
    "ProgramMode",
    "IFormattedName",
    "CmdLineArgs",
    "ProgramArgs",
    "VariableCache",
    "CheartTopology",
]
import abc
import dataclasses as dc
import enum
from ..io.indexing import SearchMode
from ..tools.basiclogging import LogLevel
import numpy as np
from typing import Final, Literal
from ..var_types import *
from ..xmlwriter.vtk_elements import (
    VtkBoundaryElement,
    VtkTopologyElement,
    get_element_type,
)


class ProgramMode(enum.StrEnum):
    none = "none"
    search = "search"
    searchsubindex = "searchsubindex"
    range = "range"
    subindex = "subindex"
    subauto = "subauto"


class IFormattedName(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, i: str | int) -> str: ...


@dc.dataclass(slots=True)
class CmdLineArgs:
    cmd: Literal["find", "index"]
    var: Final[list[str]]
    prefix: Final[str]
    input_folder: Final[str]
    output_folder: Final[str]
    time_series: Final[str | None]
    progress_bar: Final[bool]
    log: Final[LogLevel]
    binary: Final[bool]
    compression: Final[bool]
    xfile: Final[str]
    tfile: Final[str]
    bfile: Final[str | None]
    disp: Final[str | None]
    cores: Final[int] = 1
    index: tuple[int, int, int] | SearchMode = SearchMode.none
    subindex: tuple[int, int, int] | SearchMode = SearchMode.none


@dc.dataclass(slots=True)
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


class CheartTopology:
    __slots__ = ["_ft", "ne", "nc", "vtkelementtype", "vtksurfacetype"]

    _ft: Arr[tuple[int, int], i32]
    ne: int
    nc: int
    vtkelementtype: VtkTopologyElement
    vtksurfacetype: VtkBoundaryElement

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
        self.vtkelementtype, self.vtksurfacetype = get_element_type(self.nc, bfile)

    def __setitem__(self, index: int, data: Vec[i32]) -> None:
        self._ft[index] = data

    def __getitem__(self, index: int) -> int | Vec[i32]:
        return self._ft[index]

    def get_data(self) -> Mat[i32]:
        return self._ft


@dc.dataclass(slots=True)
class VariableCache:
    i0: Final[str | int]
    top: Final[CheartTopology]
    nodes: Final[Arr[tuple[int, int], f64]]
    space: str
    disp: str | None
    var: dict[str, str] = dc.field(default_factory=dict)


@dc.dataclass(slots=True)
class InputArguments:
    space: str | Arr[tuple[int, int], f64]
    disp: str | None
    var: dict[str, str]
    prefix: str
