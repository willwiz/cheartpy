import abc
from collections.abc import ValuesView
from typing import Any, TextIO

from ..aliases import *
from .basic import *

__all__ = ["ISolverMatrix"]


class ISolverMatrix(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @property
    @abc.abstractmethod
    def suppress_output(self) -> bool: ...
    @suppress_output.setter
    @abc.abstractmethod
    def suppress_output(self, val: bool): ...

    # @abc.abstractmethod
    # def get_aux_var(self) -> Sequence[_Variable]: ...
    @abc.abstractmethod
    def get_problems(self) -> ValuesView[IProblem]: ...
    @abc.abstractmethod
    def AddSetting(self, opt: str, *val: Any) -> None: ...
    @abc.abstractmethod
    def AddProblem(self, *prob: IProblem) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...
