import dataclasses as dc
from typing import TYPE_CHECKING, Literal, Required, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from cheartpy.search import SearchMode
    from pytools.arrays import DType
    from pytools.logging import LogEnum, LogLevel

SubparserModes = Literal["index", "find"]


class APIKwargsFind(TypedDict, total=False):
    index: tuple[int, int, int]
    subindex: tuple[int, int, int] | Literal["auto"]
    mesh: Required[Path | str]
    space: Path | str
    disp: str
    boundary: Path | str
    prefix: str
    input_dir: Path | str
    output_dir: Path | str
    prog_bar: bool
    log: LogLevel
    binary: bool
    compress: bool
    core: int
    thread: int
    interpreter: int
    cell_var: Sequence[str]
    point_var: Sequence[str]


class APIKwargsIndex(TypedDict, total=False):
    index: tuple[int, int, int]
    subindex: tuple[int, int, int] | Literal["auto"]
    top: Required[Path | str]
    space: Required[Path | str]
    disp: str
    boundary: Path | str
    prefix: str
    input_dir: Path | str
    output_dir: Path | str
    prog_bar: bool
    log: LogLevel
    binary: bool
    compress: bool
    core: int
    thread: int
    interpreter: int
    cell_var: Sequence[str]
    point_var: Sequence[str]


class TimeSeriesKwargs(TypedDict, total=False):
    prefix: Required[str]
    time: Required[Path | float]
    folder: Path
    log: LogLevel
    dtype: DType[np.floating]


@dc.dataclass(slots=True, frozen=True)
class VTUProgArgs:
    cmd: SubparserModes
    index: tuple[int, int, int] | SearchMode
    subindex: tuple[int, int, int] | SearchMode
    prefix: str | None
    input_dir: Path
    output_dir: Path | None
    top: Path
    space: Path | str
    disp: str | None
    boundary: Path | None
    prog_bar: bool
    log: LogEnum
    binary: bool
    compress: bool
    core: int | None
    thread: int | None
    interpreter: int | None
    cell_var: Sequence[str]
    point_var: Sequence[str]


@dc.dataclass(slots=True, frozen=True)
class TimeProgArgs:
    cmd: str
    prefix: str
    time: Path | float
    folder: Path
