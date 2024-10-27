#!/usr/bin/env python3
import dataclasses as dc
from typing import Sequence, TextIO, ValuesView
from ..aliases import *
from ..pytools import *
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
class SolverSubGroup(_SolverSubGroup):
    method: SolverSubgroupAlgorithm
    problems: dict[str, _SolverMatrix | _Problem] = dc.field(default_factory=dict)
    aux_vars: dict[str, _Variable] = dc.field(default_factory=dict)
    _scale_first_residual: float | None = None

    def __post_init__(self):
        for p in self.problems.values():
            for v in p.get_aux_vars():
                self.aux_vars[str(v)] = v

    def get_method(self) -> SolverSubgroupAlgorithm:
        return self.method

    def get_aux_vars(self) -> dict[str, _Variable]:
        return self.aux_vars

    def get_problems(self) -> ValuesView[_SolverMatrix | _Problem]:
        return self.problems.values()

    @property
    def scale_first_residual(self) -> float | None:
        return self._scale_first_residual

    @scale_first_residual.setter
    def scale_first_residual(self, value: float | None) -> None:
        self._scale_first_residual = value


@dc.dataclass(slots=True)
class SolverGroup(_SolverGroup):
    name: str
    time: _TimeScheme
    sub_groups: list[_SolverSubGroup] = dc.field(default_factory=list)
    aux_vars: dict[str, _Variable] = dc.field(default_factory=dict)
    settings: dict[
        TolSettings | IterationSettings | Literal["CatchSolverErrors"],
        list[str | int | float | _Expression | _Variable],
    ] = dc.field(default_factory=dict)
    export_initial_condition: bool = True
    use_dynamic_topologies: bool | float = False

    def __repr__(self) -> str:
        return self.name

    # TOL
    def __post_init__(self):
        for sg in self.sub_groups:
            for k, v in sg.get_aux_vars().items():
                self.aux_vars[k] = v

    def get_time_scheme(self) -> _TimeScheme:
        return self.time

    def get_aux_vars(self) -> ValuesView[_Variable]:
        return self.aux_vars.values()

    def get_subgroups(self) -> Sequence[_SolverSubGroup]:
        return self.sub_groups

    def set_convergence(
        self,
        task: TolSettings | TOL_SETTINGS,
        val: float | str,
    ) -> None:
        task = get_enum(task, TolSettings)
        self.settings[task] = [val]

    def set_iteration(
        self,
        task: IterationSettings | ITERATION_SETTINGS,
        val: int | str,
    ) -> None:
        task = get_enum(task, IterationSettings)
        self.settings[task] = [val]

    def catch_solver_errors(
        self,
        err: Literal["nan_maxval"],
        act: Literal["evaluate_full"],
        thresh: float = 1.0e10,
    ) -> None:
        self.settings["CatchSolverErrors"] = [err, act, thresh]

    def AddVariable(self, *var: _Variable):
        for v in var:
            self.aux_vars[str(v)] = v

    def RemoveVariable(self, *var: str | _Variable):
        for v in var:
            if isinstance(v, str):
                self.aux_vars.pop(v)
            else:
                self.aux_vars.pop(str(v))

    # SG
    def AddSolverSubGroup(self, *sg: _SolverSubGroup) -> None:
        for v in sg:
            self.sub_groups.append(v)
            for x in v.get_aux_vars().values():
                self.AddVariable(x)

    def RemoveSolverSubGroup(self, *sg: _SolverSubGroup) -> None:
        for v in sg:
            self.sub_groups.remove(v)

    def MakeSolverSubGroup(
        self,
        method: Literal["seq_fp_linesearch", "SOLVER_SEQUENTIAL"],
        *problems: _SolverMatrix | _Problem,
    ) -> None:
        self.sub_groups.append(
            SolverSubGroup(
                method=get_enum(method, SolverSubgroupAlgorithm),
                problems={str(p): p for p in problems},
            )
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
            f.write(f"  !SetSolverGroup={{{self.name}|export_initial_condition}}\n")
        # Print Conv Settings
        for k, v in self.settings.items():
            string = join_fields(self.name, k, *v)
            f.write(f"  !SetSolverGroup={{{string}}}\n")
        if self.use_dynamic_topologies:
            f.write(f"  !SetSolverGroup={{{self.name}|UsingDynamicTopologies}}\n")
        for g in self.sub_groups:
            pobs = [VoS(p) for p in g.get_problems()]
            if g.scale_first_residual:
                f.write(
                    f'!DefSolverSubGroup={{{self.name}|{g.get_method()}|{
                        "|".join(pobs)}|ScaleFirstResidual[{g.scale_first_residual}]}}\n'
                )
            else:
                f.write(
                    f"!DefSolverSubGroup={{{join_fields(self,
                        g.get_method(), *pobs)}}}\n"
                )
