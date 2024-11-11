__all__ = ["CheartMeshFormat", "CheartVarFormat", "CheartZipFormat"]
from typing import Final
from ..tools.path_tools import path
from ..var_types import *
from .interfaces import IFormattedName


class CheartMeshFormat(IFormattedName):
    __slots__ = ["var"]

    var: Final[str]

    def __init__(self, var: str) -> None:
        self.var = var

    def __getitem__(self, unused_time: str | int) -> str:
        return self.var


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
