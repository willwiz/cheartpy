import abc
from typing import TextIO
from ..aliases import *

__all__ = ["_Basis", "_Quadrature", "_CheartBasis"]


class _Basis(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...


class _Quadrature(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...


class _CheartBasis(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...
