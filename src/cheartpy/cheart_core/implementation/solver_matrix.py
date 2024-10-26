#!/usr/bin/env python3
import dataclasses as dc
from typing import TextIO, ValuesView

from cheartpy.cheart_core.interface.basis import _Variable
from ..interface import *
from ..aliases import *
from ..pytools import *


@dc.dataclass(slots=True)
class SolverMatrix(_SolverMatrix):
    name: str
    solver: MatrixSolverTypes
    problem: dict[str, _Problem] = dc.field(default_factory=dict)
    suppress_output: bool = True
    aux_vars: dict[str, _Variable] = dc.field(default_factory=dict)
    settings: dict[str, list[str]] = dc.field(default_factory=dict)

    def __post_init__(self):
        for _, p in self.problem.items():
            for v in p.get_aux_vars():
                self.aux_vars[str(v)] = v

    def __repr__(self) -> str:
        return self.name

    def get_aux_vars(self):
        return self.aux_vars.values()

    def get_problems(self) -> ValuesView[_Problem]:
        return self.problem.values()

    def AddSetting(self, opt, *val):
        self.settings[opt] = list(val)

    def AddProblem(self, *prob: _Problem):
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
