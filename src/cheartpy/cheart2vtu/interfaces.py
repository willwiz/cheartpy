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
from typing import Final
from ..var_types import *
from ..xmlwriter import IVtkElementInterface, get_element_type


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
    mesh: Final[str | tuple[str, str, str]]
    var: Final[list[str]]
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


class CheartTopology:
    __slots__ = ["_ft", "ne", "nc", "vtkelementtype", "vtksurfacetype"]

    _ft: Mat[int_t]
    ne: int
    nc: int
    vtkelementtype: type[IVtkElementInterface]
    vtksurfacetype: type[IVtkElementInterface]

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

    def __setitem__(self, index: int, data: Vec[int_t]) -> None:
        self._ft[index] = data

    def __getitem__(self, index: int) -> int | Vec[int_t]:
        return self._ft[index]

    def get_data(self) -> Mat[int_t]:
        return self._ft


@dc.dataclass(slots=True)
class VariableCache:
    top: Final[CheartTopology]
    t: str | int
    space_i: str
    disp_i: str | None
    space: Mat[f64]
    disp: Mat[f64]
    x: Mat[f64]
    var_i: dict[str, str] = dc.field(default_factory=dict)
    var: dict[str, Mat[f64]] = dc.field(default_factory=dict)


@dc.dataclass(slots=True)
class InputArguments:
    space: str | Arr[tuple[int, int], f64]
    disp: str | None
    var: dict[str, str]
    prefix: str
