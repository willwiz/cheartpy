__all__ = ["CheartMeshFormat", "CheartVarFormat", "CheartZipFormat"]
from pathlib import Path
from typing import Final

from .trait import IFormattedName


class CheartMeshFormat(IFormattedName):
    __slots__ = ["folder", "var"]

    folder: Final[Path]
    var: Final[str]

    def __init__(self, folder: Path | str | None, var: str) -> None:
        self.folder = Path(folder) if folder else Path()
        self.var = var

    def __getitem__(self, _time: str | int) -> Path:
        return self.folder / self.var

    @property
    def name(self) -> str:
        return self.var


class CheartVarFormat(IFormattedName):
    __slots__ = ["folder", "var"]

    folder: Final[Path]
    var: Final[str]

    def __init__(self, folder: Path | str | None, var: str) -> None:
        self.folder = Path(folder) if folder else Path()
        self.var = var

    def __getitem__(self, time: str | int) -> Path:
        return self.folder / f"{self.var}-{time}.D"

    @property
    def name(self) -> str:
        return self.var


class CheartZipFormat(IFormattedName):
    __slots__ = ["folder", "var"]

    folder: Final[Path]
    var: Final[str]

    def __init__(self, folder: Path | str | None, var: str) -> None:
        self.folder = Path(folder) if folder else Path()
        self.var = var

    def __getitem__(self, time: str | int) -> Path:
        return self.folder / f"{self.var}-{time}.D.gz"

    @property
    def name(self) -> str:
        return self.var


class CheartVTUFormat(IFormattedName):
    __slots__ = ["folder", "prefix"]

    folder: Final[Path]
    prefix: Final[str]

    def __init__(self, folder: Path | str | None, var: str) -> None:
        self.folder = Path(folder) if folder else Path()
        self.prefix = var

    def __getitem__(self, time: str | int) -> Path:
        return self.folder / f"{self.prefix}-{time}.vtu"

    @property
    def name(self) -> str:
        return self.prefix
