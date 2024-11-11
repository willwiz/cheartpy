__all__ = ["SolverMatrix"]
import dataclasses as dc
from typing import TextIO, ValuesView
from ..interface import *
from ..aliases import *
from ..pytools import *


@dc.dataclass(slots=True)
class SolverMatrix(ISolverMatrix):
    name: str
    solver: MatrixSolverTypes
    problem: dict[str, IProblem] = dc.field(default_factory=dict)
    _suppress_output: bool = dc.field(default=True)
    settings: dict[str, list[str]] = dc.field(default_factory=dict)

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
    def suppress_output(self, val: bool):
        self._suppress_output = val

    # def get_aux_var(self):
    #     return [v for p in self.problem.values() for v in p.get_var_deps()]

    def get_problems(self) -> ValuesView[IProblem]:
        return self.problem.values()

    def AddSetting(self, opt: str, *val: Any):
        self.settings[opt] = list(val)

    def AddProblem(self, *prob: IProblem):
        for p in prob:
            self.problem[str(p)] = p

    def write(self, f: TextIO):
        string = join_fields(self.name, self.solver, *self.problem.values())
        f.write(f"!DefSolverMatrix={{{string}}}\n")
        if self.suppress_output:
            f.write(f"  !SetSolverMatrix={{{self.name}|SuppressOutput}}\n")
        for k, v in self.settings.items():
            string = join_fields(self.name, k, *v)
            f.write(f"  !SetSolverMatrix={{{string}}}\n")
