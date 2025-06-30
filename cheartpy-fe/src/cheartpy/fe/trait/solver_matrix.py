import abc
from collections.abc import ValuesView
from typing import TextIO

from .basic import ICheartTopology, IExpression, IProblem

__all__ = ["ISolverMatrix"]


class ISolverMatrix(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @property
    @abc.abstractmethod
    def suppress_output(self) -> bool: ...
    @suppress_output.setter
    @abc.abstractmethod
    def suppress_output(self, val: bool) -> None: ...

    # @abc.abstractmethod
    # def get_aux_var(self) -> Sequence[_Variable]: ...
    @abc.abstractmethod
    def get_problems(self) -> ValuesView[IProblem]: ...
    @abc.abstractmethod
    def add_setting(
        self,
        opt: str,
        *val: str | int | IExpression | tuple[ICheartTopology, int],
    ) -> None: ...
    @abc.abstractmethod
    def add_problem(self, *prob: IProblem) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...
