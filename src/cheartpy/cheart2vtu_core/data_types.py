import dataclasses as dc
import enum
import os
import numpy as np
from typing import Final, Literal
from cheartpy.types import i32, f64, Arr
from cheartpy.xmlwriter.vtk_elements import (
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


@dc.dataclass(slots=True)
class CmdLineArgs:
    cmd: Literal["find", "index"]
    var: Final[list[str]]
    prefix: Final[str]
    input_folder: Final[str]
    output_folder: Final[str]
    time_series: Final[str | None]
    progress_bar: Final[bool]
    verbose: Final[bool]
    binary: Final[bool]
    compression: Final[bool]
    cores: Final[int]
    xfile: Final[str]
    tfile: Final[str]
    bfile: Final[str | None]
    disp: Final[str | None]
    step: int | None = None
    index: tuple[int, int, int] | None = None
    sub_index: tuple[int, int, int] | None = None
    sub_auto: bool = False


class CheartMeshFormat:
    __slots__ = ["var"]

    var: Final[str]

    def __init__(self, var: str) -> None:
        self.var = var

    def get_name(self, unused_time: str | int) -> str:
        return self.var


class CheartVarFormat:
    __slots__ = ["folder", "var"]

    folder: Final[str]
    var: Final[str]

    def __init__(self, folder: str, var: str) -> None:
        self.folder = folder
        self.var = var

    def get_name(self, time: str | int) -> str:
        return os.path.join(self.folder, f"{self.var}-{time}.D")


class CheartZipFormat:
    __slots__ = ["folder", "var"]

    folder: Final[str]
    var: Final[str]

    def __init__(self, folder: str, var: str) -> None:
        self.folder = folder
        self.var = var

    def get_name(self, time: str | int) -> str:
        return os.path.join(self.folder, f"{self.var}-{time}.D.gz")


@dc.dataclass(slots=True)
class ProgramArgs:
    cmd: ProgramMode
    prefix: Final[str]
    input_folder: Final[str]
    output_folder: Final[str]
    time_series: Final[str | None]
    progress_bar: Final[bool]
    verbose: Final[bool]
    binary: Final[bool]
    compression: Final[bool]
    cores: Final[int]
    tfile: Final[str]
    bfile: Final[str | None]
    space: Final[CheartMeshFormat | CheartVarFormat | CheartZipFormat]
    disp: Final[CheartMeshFormat | CheartVarFormat | CheartZipFormat | None]
    var: Final[dict[str, CheartVarFormat | CheartZipFormat]]


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
        self.vtkelementtype, self.vtksurfacetype = get_element_type(
            self.nc, bfile)

    def __setitem__(self, index, data) -> None:
        self._ft[index] = data

    def __getitem__(self, index) -> int | Arr[int, i32]:
        return self._ft[index]


@dc.dataclass(slots=True)
class VariableCache:
    i0: Final[str]
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
