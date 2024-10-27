import abc
from typing import TextIO, ValuesView
from ..aliases import *
from .basis import *

__all__ = ["_SolverMatrix"]


class _SolverMatrix(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @property
    @abc.abstractmethod
    def suppress_output(self) -> bool: ...
    @suppress_output.setter
    @abc.abstractmethod
    def suppress_output(self, val: bool): ...
    @abc.abstractmethod
    def get_aux_vars(self) -> ValuesView[_Variable]: ...
    @abc.abstractmethod
    def get_problems(self) -> ValuesView[_Problem]: ...
    @abc.abstractmethod
    def AddSetting(self, opt, *val) -> None: ...
    @abc.abstractmethod
    def AddProblem(self, *prob: _Problem) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...
