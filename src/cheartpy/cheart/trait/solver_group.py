import abc
from typing import Mapping, Sequence, TextIO, ValuesView
from ..aliases import *
from .basic import *
from .solver_matrix import *

__all__ = ["ISolverSubGroup", "ISolverGroup"]


class ISolverSubGroup(abc.ABC):
    @abc.abstractmethod
    def get_method(self) -> SolverSubgroupAlgorithm: ...
    @abc.abstractmethod
    def get_all_vars(self) -> Mapping[str, IVariable]: ...
    @abc.abstractmethod
    def get_prob_vars(self) -> Mapping[str, IVariable]: ...
    @abc.abstractmethod
    def get_systems(self) -> ValuesView[IProblem | ISolverMatrix]: ...
    @abc.abstractmethod
    def get_problems(self) -> Sequence[IProblem]: ...
    @abc.abstractmethod
    def get_matrices(self) -> Sequence[ISolverMatrix]: ...
    @property
    @abc.abstractmethod
    def scale_first_residual(self) -> float | None: ...
    @scale_first_residual.setter
    @abc.abstractmethod
    def scale_first_residual(self, value: float | None): ...


class ISolverGroup(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @property
    @abc.abstractmethod
    def export_initial_condition(self) -> bool: ...
    @export_initial_condition.setter
    @abc.abstractmethod
    def export_initial_condition(self, value: bool): ...
    @abc.abstractmethod
    def get_time_scheme(self) -> ITimeScheme: ...
    @abc.abstractmethod
    def get_aux_vars(self) -> ValuesView[IVariable]: ...
    @abc.abstractmethod
    def get_subgroups(self) -> Sequence[ISolverSubGroup]: ...
    @abc.abstractmethod
    def set_convergence(
        self,
        task: TolSettings | TOL_SETTINGS,
        val: float | str,
    ) -> None: ...
    @abc.abstractmethod
    def set_iteration(
        self,
        task: IterationSettings | ITERATION_SETTINGS,
        val: int | str,
    ) -> None: ...
    @abc.abstractmethod
    def catch_solver_errors(
        self, err: Literal["nan_maxval"], act: Literal["evaluate_full"]
    ) -> None: ...
    @abc.abstractmethod
    def AddAuxVariable(self, *var: IVariable): ...
    @abc.abstractmethod
    def RemoveAuxVariable(self, *var: str | IVariable): ...

    # SG
    @abc.abstractmethod
    def AddSolverSubGroup(self, *sg: ISolverSubGroup) -> None: ...
    @abc.abstractmethod
    def RemoveSolverSubGroup(self, *sg: ISolverSubGroup) -> None: ...
    @abc.abstractmethod
    def MakeSolverSubGroup(
        self,
        method: Literal["seq_fp_linesearch", "SOLVER_SEQUENTIAL"],
        *problems: ISolverMatrix | IProblem,
    ) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...
