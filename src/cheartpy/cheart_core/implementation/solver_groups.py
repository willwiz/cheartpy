#!/usr/bin/env python3
import dataclasses as dc
from typing import TextIO, overload
from ..aliases import *
from ..pytools import *
from ..interface.basis import *
from .variables import Variable
from .expressions import Expression
from .time_schemes import TimeScheme
from .solver_matrices import SolverMatrix
from ..interface import *

"""
Cheart dataclasses

Structure:

PFile : {TimeScheme, SolverGroup, SolverMatrix, Basis, Topology, TopInterface,
         Variable, Problem, Expressions}

Topology -> TopInterface

BCPatch -> BoundaryCondtion
Matlaw -> SolidProblem(Problem)
BoundaryCondtion -> Problem


TimeScheme -> SolverGroup
SolverSubgroup -> SolverGroup
Problem -> SolverMatrix -> SolverSubgroup
Problem -> SolverSubgroup (method: SOLVER_SEQUENTIAL)


Content:

TimeScheme
Basis
Topology
TopInterface
Variable
BCPatch
BoundaryCondtion
Matlaw (SolidProblem)
Problem : {SolidProblem}
SolverMatrix
SolverSubgroup
SolverGroup
Expressions
PFile
"""


# Define Solver SubGroup
@dc.dataclass(slots=True)
class SolverSubGroup:
    method: SolverSubgroupAlgorithm
    problems: dict[str, SolverMatrix | _Problem] = dc.field(
        default_factory=dict
    )
    aux_vars: dict[str, _Variable] = dc.field(default_factory=dict)
    scale_file_residual: bool = False

    def __post_init__(self):
        for p in self.problems.values():
            for k, x in p.get_aux_vars().items():
                self.aux_vars[k] = x


@dc.dataclass
class SolverGroup(object):
    name: str
    time: _TimeScheme
    SolverSubGroups: list[SolverSubGroup] = dc.field(default_factory=list)
    aux_vars: dict[str, _Variable] = dc.field(default_factory=dict)
    settings: dict[str, list[str | int | float]
                   ] = dc.field(default_factory=dict)
    export_initial_condition: bool = False
    use_dynamic_topologies: bool | float = False

    # TOL
    def __post_init__(self):
        for sg in self.SolverSubGroups:
            for k, v in sg.aux_vars.items():
                self.aux_vars[k] = v

    @overload
    def AddSetting(self,
        task: Literal[
            "L2TOL",
            "L2PERCENT",
            "INFRES",
            "INFUPDATE",
            "INFDEL",
            "INFRELUPDATE",
            "L2RESRELPERCENT",
        ] | Literal["ITERATION", "SUBITERATION", "LINESEARCHITER", "SUBITERFRACTION"],
        val: _Expression| _Variable| float| str,
    ) -> None:
        ...

    @overload
    def AddSetting(self, task: TolSettings, val: float | str) -> None:
        ...

    @overload
    def AddSetting(self, task: IterationSettings, val: int | str) -> None:
        ...

    def AddSetting(self, task, val):
        self.settings[task] = [val]

    # VAR
    def set_convergence(
        self,
        task: TolSettings
        | Literal[
            "L2TOL",
            "L2PERCENT",
            "INFRES",
            "INFUPDATE",
            "INFDEL",
            "INFRELUPDATE",
            "L2RESRELPERCENT",
        ],
        val: float|str,
    ) -> None:
        self.settings[task] = [val]

    def set_iteration(
        self,
        task: IterationSettings
        | Literal[
            "ITERATION",
            "SUBITERATION",
            "LINESEARCHITER",
            "SUBITERFRACTION",
            "GroupIterations",
        ],
        val: int|str,
    ) -> None:
        self.settings[task] = [val]

    def catch_solver_errors(
        self, err: Literal["nan_maxval"], act: Literal["evaluate_full"]
    ) -> None:
        self.settings["CatchSolverErrors"] = [err, act]

    def AddVariable(self, *var: _Variable):
        for v in var:
            self.aux_vars[str(v)] = v


    def RemoveVariable(self, *var: str|Variable):
        for v in var:
            if isinstance(v, str):
                self.aux_vars.pop(v)
            else:
                self.aux_vars.pop(v.name)

    # SG
    def AddSolverSubGroup(self, *sg: SolverSubGroup) -> None:
        for v in sg:
            self.SolverSubGroups.append(v)
            for x in v.aux_vars.values():
                self.AddVariable(x)

    def RemoveSolverSubGroup(self, *sg: SolverSubGroup) -> None:
        for v in sg:
            self.SolverSubGroups.remove(v)

    def MakeSolverSubGroup(
        self,
        method: Literal["seq_fp_linesearch", "SOLVER_SEQUENTIAL"],
        *problems: SolverMatrix| _Problem,
    ) -> None:
        self.SolverSubGroups.append(
            SolverSubGroup(method=get_enum(method, SolverSubgroupAlgorithm),
                           problems={repr(p): p for p in problems})
        )

    # WRITE
    def write(self, f: TextIO) -> None:
        # if isinstance(self.time,TimeScheme):
        #   self.time.write(f)
        f.write(hline("Solver Groups"))
        f.write(f"!DefSolverGroup={{{self.name}|{VoS(self.time)}}}\n")
        # Handle Additional Vars
        vars = [VoS(v) for v in self.aux_vars.values()]
        for l in splicegen(45, vars):
            if l:
                f.write(
                    f'  !SetSolverGroup={{{
                        self.name}|AddVariables|{"|".join(l)}}}\n'
                )
        # Print export init setting
        if self.export_initial_condition:
            f.write(
                f"  !SetSolverGroup={{{self.name}|export_initial_condition}}\n")
        # Print Conv Settings
        for k, v in self.settings.items():
            string = join_fields(self.name, k, *v)
            f.write(f"  !SetSolverGroup={{{string}}}\n")
        if self.use_dynamic_topologies:
            f.write(
                f"  !SetSolverGroup={{{self.name}|UsingDynamicTopologies}}\n")
        for g in self.SolverSubGroups:
            pobs = [VoS(p) for p in g.problems]
            if g.scale_file_residual:
                f.write(
                    f'!DefSolverSubGroup={{{self.name}|{g.method}|{
                        "|".join(pobs)}|ScaleFirstResidual[{g.scale_file_residual}]}}\n'
                )
            else:
                f.write(
                    f'!DefSolverSubGroup={{{self.name}|{
                        g.method}|{"|".join(pobs)}}}\n'
                )


def hash_tops(tops: list[_CheartTopology] | list[str]) -> str:
    names = [VoS(t) for t in tops]
    return "_".join(names)
