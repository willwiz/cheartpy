import abc
from typing import Mapping, Sequence, TextIO, ValuesView
from ..aliases import *
from .basis import *
from .solver_matrix import *

__all__ = ["_SolverSubGroup", "_SolverGroup"]


class _SolverSubGroup(abc.ABC):
    @abc.abstractmethod
    def get_method(self) -> SolverSubgroupAlgorithm: ...
    @abc.abstractmethod
    def get_all_vars(self) -> Mapping[str, _Variable]: ...
    @abc.abstractmethod
    def get_prob_vars(self) -> Mapping[str, _Variable]: ...
    @abc.abstractmethod
    def get_systems(self) -> ValuesView[_Problem | _SolverMatrix]: ...
    @abc.abstractmethod
    def get_problems(self) -> Sequence[_Problem]: ...
    @abc.abstractmethod
    def get_matrices(self) -> Sequence[_SolverMatrix]: ...
    @property
    @abc.abstractmethod
    def scale_first_residual(self) -> float | None: ...
    @scale_first_residual.setter
    @abc.abstractmethod
    def scale_first_residual(self, value: float | None): ...


class _SolverGroup(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def get_time_scheme(self) -> _TimeScheme: ...
    @abc.abstractmethod
    def get_aux_vars(self) -> ValuesView[_Variable]: ...
    @abc.abstractmethod
    def get_subgroups(self) -> Sequence[_SolverSubGroup]: ...
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
    def AddAuxVariable(self, *var: _Variable): ...
    @abc.abstractmethod
    def RemoveAuxVariable(self, *var: str | _Variable): ...

    # SG
    @abc.abstractmethod
    def AddSolverSubGroup(self, *sg: _SolverSubGroup) -> None: ...
    @abc.abstractmethod
    def RemoveSolverSubGroup(self, *sg: _SolverSubGroup) -> None: ...
    @abc.abstractmethod
    def MakeSolverSubGroup(
        self,
        method: Literal["seq_fp_linesearch", "SOLVER_SEQUENTIAL"],
        *problems: _SolverMatrix | _Problem,
    ) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...
