from pathlib import Path
from typing import Final

from ._trait import IFormattedName


class CheartFileFormat(IFormattedName):
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


class CheartResFormat(IFormattedName):
    __slots__ = ["folder", "var"]

    folder: Final[Path]
    var: Final[str]

    def __init__(self, folder: Path | str | None, var: str) -> None:
        self.folder = Path(folder) if folder else Path()
        self.var = var

    def __getitem__(self, time: str | int) -> Path:
        return self.folder / f"{self.var}-{time}.res2"

    @property
    def name(self) -> str:
        return self.var


class CheartVTUFormat:
    __slots__ = ("count", "folder", "prefix")

    count: int
    folder: Final[Path]
    prefix: Final[str]

    def __init__(
        self, folder: Path | str | None, var: str, *, assume_subindex: bool = False
    ) -> None:
        self.folder = Path(folder) if folder else Path()
        self.prefix = var
        self.count = 0 if assume_subindex else -1

    def __getitem__(self, time: int | tuple[int, int]) -> Path:
        if self.count < 0:
            return self.folder / f"{self.prefix}-{time}.vtu"
        name = self.folder / f"{self.prefix}-{self.count}.{time}.vtu"
        self.count = self.count + 1
        return name

    @property
    def name(self) -> str:
        return self.prefix
