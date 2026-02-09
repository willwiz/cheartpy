import dataclasses as dc
from typing import TYPE_CHECKING, TextIO

from cheartpy.fe.trait import ICheartTopology, IExpression, IProblem, ISolverMatrix
from cheartpy.fe.utils import join_fields

if TYPE_CHECKING:
    from collections.abc import ValuesView

    from cheartpy.fe.aliases import (
        MatrixSolverEnum,
    )

__all__ = ["SolverMatrix"]

_REMAINING_LINE_LEN = 45


@dc.dataclass(slots=True)
class SolverMatrix(ISolverMatrix):
    name: str
    solver: MatrixSolverEnum
    problem: dict[str, IProblem] = dc.field(default_factory=dict[str, IProblem])
    _suppress_output: bool = dc.field(default=True)
    settings: dict[str, list[str]] = dc.field(default_factory=dict[str, list[str]])

    # def __post_init__(self):
    #     for _, p in self.problem.items():
    #         for v in p.get_aux_vars():
    #             self.aux_vars[str(v)] = v

    def __repr__(self) -> str:
        return self.name

    @property
    def suppress_output(self) -> bool:
        return self._suppress_output

    @suppress_output.setter
    def suppress_output(self, val: bool) -> None:
        self._suppress_output = val

    # def get_aux_var(self):
    #     return [v for p in self.problem.values() for v in p.get_var_deps()]

    def get_problems(self) -> ValuesView[IProblem]:
        return self.problem.values()

    def add_setting(
        self,
        opt: str,
        *val: str | int | IExpression | tuple[ICheartTopology, int],
    ) -> None:
        self.settings[opt] = [str(v) for v in val]

    def add_problem(self, *prob: IProblem) -> None:
        for p in prob:
            self.problem[str(p)] = p

    def write(self, f: TextIO) -> None:
        _solver_matrix_write(self, f)


def _solver_matrix_write(matrix: SolverMatrix, f: TextIO) -> None:
    string = join_fields(matrix.name, matrix.solver, *matrix.problem.values())
    if len(string) > _REMAINING_LINE_LEN:
        f.write(f"!DefSolverMatrix={{{join_fields(matrix.name, matrix.solver)}}}\n")
        f.writelines(f"    {v}\n" for v in matrix.problem.values())
    else:
        f.write(f"!DefSolverMatrix={{{string}}}\n")
    if matrix.suppress_output:
        f.write(f"  !SetSolverMatrix={{{matrix.name}|SuppressOutput}}\n")
    for k, v in matrix.settings.items():
        string = join_fields(matrix.name, k, *v)
        f.write(f"  !SetSolverMatrix={{{string}}}\n")
