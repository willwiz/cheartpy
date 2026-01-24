import dataclasses as dc
from typing import TYPE_CHECKING, Final, Literal, Required, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from cheartpy.search.trait import SearchMode
    from pytools.arrays import DType
    from pytools.logging import LOG_LEVEL, LogLevel

SUBPARSER_MODES = Literal["index", "find"]


class APIKwargs(TypedDict, total=False):
    index: tuple[int, int, int] | None
    subindex: tuple[int, int, int] | Literal["auto", "none"]
    mesh: Path | str
    prefix: str | None
    input_dir: Path | str
    output_dir: Path | str | None
    top: Path | str
    space: Path | str | None
    boundary: Path | str | None
    prog_bar: bool
    log: LOG_LEVEL
    binary: bool
    compress: bool
    cores: int
    var: Sequence[str]


class APIKwargsFind(TypedDict, total=False):
    index: tuple[int, int, int] | None
    subindex: tuple[int, int, int] | Literal["auto", "none"]
    mesh: Required[Path | str]
    prefix: str | None
    input_dir: Path | str
    output_dir: Path | str | None
    space: Path | str | None
    boundary: Path | str | None
    prog_bar: bool
    log: LOG_LEVEL
    binary: bool
    compress: bool
    cores: int
    var: Sequence[str]


class APIKwargsIndex(TypedDict, total=False):
    index: tuple[int, int, int] | None
    subindex: tuple[int, int, int] | Literal["auto", "none"]
    top: Required[Path | str]
    prefix: str | None
    input_dir: Path | str
    output_dir: Path | str | None
    space: Required[Path | str]
    boundary: Path | str | None
    prog_bar: bool
    log: LOG_LEVEL
    binary: bool
    compress: bool
    cores: int
    var: Sequence[str]


class TimeSeriesKwargs(TypedDict, total=False):
    prefix: Required[str]
    time: Required[Path | float]
    root: Path
    log: LOG_LEVEL
    dtype: DType[np.floating]


@dc.dataclass(slots=True)
class VTUProgArgs:
    cmd: Final[SUBPARSER_MODES]
    index: tuple[int, int, int] | SearchMode | None
    subindex: tuple[int, int, int] | SearchMode | None
    prefix: Final[str | None]
    input_dir: Final[Path]
    output_dir: Final[Path | None]
    mesh_or_top: Final[Path]
    space: Final[Path | None]
    boundary: Final[Path | None]
    prog_bar: Final[bool]
    log: Final[LogLevel]
    binary: Final[bool]
    compress: Final[bool]
    cores: Final[int]
    var: Final[Sequence[str]]


@dc.dataclass(slots=True)
class TimeProgArgs:
    cmd: Final[str]
    prefix: str
    time: Path | float
    root: Path
