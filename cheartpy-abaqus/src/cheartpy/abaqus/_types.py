from typing import TYPE_CHECKING, Required, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from pytools.logging import LogLevel


class AbaqusAPIKwargs(TypedDict, total=False):
    """Keyword arguments for Abaqus Converter API functions."""

    files: Required[Sequence[Path | str]]
    topology: Required[Sequence[str]]
    boundary: Mapping[int, Sequence[str]] | None
    masks: Mapping[str, tuple[str, Sequence[str]]] | None
    prefix: str | None
    log_level: LogLevel
    cores: int
