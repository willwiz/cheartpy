#!/usr/bin/env python3
import dataclasses as dc
from typing import TextIO
from .aliases import *
from .pytools import *
from .problems import *


@dc.dataclass(slots=True)
class SolverMatrix:
    name: str
    solver: MatrixSolverTypes
    problem: dict[str, Problem] = dc.field(default_factory=dict)
    suppress_output: bool = True
    aux_vars: dict[str, Variable] = dc.field(default_factory=dict)
    settings: dict[str, list[str]] = dc.field(default_factory=dict)

    def __post_init__(self):
        for _, p in self.problem.items():
            for k, x in p.aux_vars.items():
                self.aux_vars[k] = x

    def AddSetting(self, opt, *val):
        self.settings[opt] = list(val)

    def AddProblem(self, *prob: Problem):
        for p in prob:
            self.problem[p.name] = p

    def write(self, f: TextIO):
        f.write(
            f'!DefSolverMatrix={{{self.name}|{self.solver}|{
                "|".join([p.name for p in self.problem.values()])}}}\n'
        )
        if self.suppress_output:
            f.write(f" !SetSolverMatrix={{{self.name}|SuppressOutput}}\n")
        for k, v in self.settings.items():
            string = join_fields([self.name, k, *v])
            f.write(f" !SetSolverMatrix={{{string}}}\n")
