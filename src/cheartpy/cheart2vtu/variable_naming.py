__all__ = ["CheartMeshFormat", "CheartVarFormat", "CheartZipFormat"]
from typing import Final
from ..tools.path_tools import path
from ..var_types import *
from .interfaces import IFormattedName


class CheartMeshFormat(IFormattedName):
    __slots__ = ["folder", "var"]

    folder: Final[str | None]
    var: Final[str]

    def __init__(self, folder: str | None, var: str) -> None:
        self.folder = folder
        self.var = var

    def __getitem__(self, unused_time: str | int) -> str:
        return path(self.folder, self.var)


class CheartVarFormat(IFormattedName):
    __slots__ = ["folder", "var"]

    folder: Final[str]
    var: Final[str]

    def __init__(self, folder: str, var: str) -> None:
        self.folder = folder
        self.var = var

    def __getitem__(self, time: str | int) -> str:
        return path(self.folder, f"{self.var}-{time}.D")


class CheartZipFormat(IFormattedName):
    __slots__ = ["folder", "var"]

    folder: Final[str]
    var: Final[str]

    def __init__(self, folder: str, var: str) -> None:
        self.folder = folder
        self.var = var

    def __getitem__(self, time: str | int) -> str:
        return path(self.folder, f"{self.var}-{time}.D.gz")


class CheartVTUFormat(IFormattedName):
    __slots__ = ["folder", "prefix"]

    folder: Final[str]
    prefix: Final[str]

    def __init__(self, folder: str, var: str) -> None:
        self.folder = folder
        self.prefix = var

    def __getitem__(self, time: str | int) -> str:
        return path(self.folder, f"{self.prefix}-{time}.vtu")
