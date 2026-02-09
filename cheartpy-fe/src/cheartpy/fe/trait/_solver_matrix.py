import abc
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from collections.abc import ValuesView

    from ._basic import ICheartTopology, IExpression, IProblem


class ISolverMatrix(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def __hash__(self) -> int: ...
    @property
    @abc.abstractmethod
    def suppress_output(self) -> bool: ...
    @suppress_output.setter
    @abc.abstractmethod
    def suppress_output(self, val: bool) -> None: ...
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
