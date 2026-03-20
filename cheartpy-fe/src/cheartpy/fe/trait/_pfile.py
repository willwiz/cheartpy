import abc
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from pathlib import Path

    from ._basic import (
        ICheartBasis,
        ICheartTopology,
        IDataPointer,
        IExpression,
        IProblem,
        ITimeScheme,
        ITopInterface,
        IVariable,
    )
    from ._solver_group import ISolverGroup, ISolverSubGroup
    from ._solver_matrix import ISolverMatrix


class IPFile(abc.ABC):
    h: str
    output_dir: Path | None

    @abc.abstractmethod
    def set_outputpath(self, path: Path | str) -> None: ...
    @abc.abstractmethod
    def add_timescheme(self, *time: ITimeScheme) -> None: ...
    @abc.abstractmethod
    def add_datapointer(self, *var: IDataPointer) -> None: ...
    @abc.abstractmethod
    def add_basis(self, *basis: ICheartBasis | None) -> None: ...
    @abc.abstractmethod
    def add_topology(self, *top: ICheartTopology) -> None: ...
    @abc.abstractmethod
    def add_interface(self, *interfaces: ITopInterface) -> None: ...
    @abc.abstractmethod
    def add_variable(self, *var: IVariable) -> None: ...
    @abc.abstractmethod
    def add_expression(self, *expr: IExpression) -> None: ...
    @abc.abstractmethod
    def add_problem(self, *prob: IProblem) -> None: ...
    @abc.abstractmethod
    def add_matrix(self, *mat: ISolverMatrix) -> None: ...
    @abc.abstractmethod
    def add_solversubgroup(self, *subgroup: ISolverSubGroup) -> None: ...
    @abc.abstractmethod
    def add_solvergroup(self, *grp: ISolverGroup) -> None: ...
    @abc.abstractmethod
    def set_exportfrequency(self, *var: IVariable, freq: int = 1) -> None: ...
    @abc.abstractmethod
    def resolve(self) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...
