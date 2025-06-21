from __future__ import annotations

__all__ = ["CheartMeshFormat", "CheartVarFormat", "CheartZipFormat"]
from pathlib import Path
from typing import Final

from .trait import IFormattedName


class CheartMeshFormat(IFormattedName):
    __slots__ = ["folder", "var"]

    folder: Final[Path]
    var: Final[str]

    def __init__(self, folder: str | None, var: str) -> None:
        self.folder = Path(folder) if folder else Path()
        self.var = var

    def __getitem__(self, _time: str | int) -> Path:
        return self.folder / self.var


class CheartVarFormat(IFormattedName):
    __slots__ = ["folder", "var"]

    folder: Final[Path]
    var: Final[str]

    def __init__(self, folder: str | None, var: str) -> None:
        self.folder = Path(folder) if folder else Path()
        self.var = var

    def __getitem__(self, time: str | int) -> Path:
        return self.folder / f"{self.var}-{time}.D"


class CheartZipFormat(IFormattedName):
    __slots__ = ["folder", "var"]

    folder: Final[Path]
    var: Final[str]

    def __init__(self, folder: str | None, var: str) -> None:
        self.folder = Path(folder) if folder else Path()
        self.var = var

    def __getitem__(self, time: str | int) -> Path:
        return self.folder / f"{self.var}-{time}.D.gz"


class CheartVTUFormat(IFormattedName):
    __slots__ = ["folder", "prefix"]

    folder: Final[Path]
    prefix: Final[str]

    def __init__(self, folder: str | None, var: str) -> None:
        self.folder = Path(folder) if folder else Path()
        self.prefix = var

    def __getitem__(self, time: str | int) -> Path:
        return self.folder / f"{self.prefix}-{time}.vtu"
