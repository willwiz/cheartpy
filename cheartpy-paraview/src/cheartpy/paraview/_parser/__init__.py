import dataclasses as dc
from typing import TYPE_CHECKING, Final, Literal, Required, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cheartpy.search.trait import SearchMode
    from pytools.logging.trait import LOG_LEVEL, LogLevel

SUBPARSER_MODES = Literal["index", "find"]


class APIKwargs(TypedDict, total=False):
    index: tuple[int, int, int] | None
    subindex: tuple[int, int, int] | Literal["auto", "none"]
    prefix: str | None
    input_dir: Path | str
    output_dir: str
    mesh: str
    top: str
    space: str | None
    boundary: Path | None
    prog_bar: bool
    log: LOG_LEVEL
    binary: bool
    compress: bool
    cores: int
    var: Sequence[str]


class APIKwargsFind(TypedDict, total=False):
    index: tuple[int, int, int] | None
    subindex: tuple[int, int, int] | Literal["auto", "none"]
    prefix: str | None
    input_dir: Path | str
    output_dir: str
    mesh: Required[str]
    space: str | None
    boundary: Path | None
    prog_bar: bool
    log: LOG_LEVEL
    binary: bool
    compress: bool
    cores: int
    var: Sequence[str]


class APIKwargsIndex(TypedDict, total=False):
    index: tuple[int, int, int] | None
    subindex: tuple[int, int, int] | Literal["auto", "none"]
    prefix: str | None
    input_dir: Path | str
    output_dir: str
    top: Required[str]
    space: Required[str]
    boundary: Path | None
    prog_bar: bool
    log: LOG_LEVEL
    binary: bool
    compress: bool
    cores: int
    var: Sequence[str]


@dc.dataclass(slots=True)
class CmdLineArgs:
    cmd: Final[SUBPARSER_MODES]
    index: tuple[int, int, int] | SearchMode | None
    subindex: tuple[int, int, int] | SearchMode | None
    prefix: Final[str | None]
    input_dir: Final[Path]
    output_dir: Final[str]
    mesh_or_top: Final[str]
    space: Final[str | None]
    boundary: Final[Path | None]
    prog_bar: Final[bool]
    log: Final[LogLevel]
    binary: Final[bool]
    compress: Final[bool]
    cores: Final[int]
    var: Final[Sequence[str]]
