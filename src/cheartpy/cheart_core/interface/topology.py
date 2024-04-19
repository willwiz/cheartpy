import abc
from typing import TextIO
from ..aliases import *

__all__ = ["_CheartTopology", "_TopInterface"]


class _CheartTopology(abc.ABC):

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...

    @abc.abstractmethod
    def AddSetting(
        self,
        task: CheartTopologySetting,
        val: int | tuple["_CheartTopology", int] | None = None,
    ): ...


class _TopInterface(abc.ABC):
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...
