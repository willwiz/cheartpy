import abc
from typing import TextIO
from ..aliases import *
from .basis import *

__all__ = ["_SolverMatrix"]


class _SolverMatrix(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def get_aux_vars(self) -> dict[str, _Variable]: ...
    @abc.abstractmethod
    def AddSetting(self, opt, *val) -> None: ...
    @abc.abstractmethod
    def AddProblem(self, *prob: _Problem) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...
