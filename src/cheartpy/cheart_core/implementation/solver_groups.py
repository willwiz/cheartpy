#!/usr/bin/env python3
from collections import ChainMap
import dataclasses as dc
from typing import Mapping, Sequence, TextIO, ValuesView

from cheartpy.cheart_core.interface.basis import _Problem
from cheartpy.cheart_core.interface.solver_matrix import _SolverMatrix
from ..aliases import *
from ..pytools import *
from ..interface import *
from .tools import recurse_get_var_list_var

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
    # aux_vars: dict[str, _Variable] = dc.field(default_factory=dict)
    _scale_first_residual: float | None = None

    # def __post_init__(self):
    #     for p in self.problems.values():
    #         for v in p.get_aux_var():
    #             self.aux_vars[str(v)] = v

    def get_method(self) -> SolverSubgroupAlgorithm:
        return self.method

    def get_all_vars(self) -> Mapping[str, _Variable]:
        _prob_vars = {str(v): v for p in self.get_problems() for v in p.get_var_deps()}
        _matrix_vars = {
            str(v): v
            for m in self.get_matrices()
            for p in m.get_problems()
            for v in p.get_var_deps()
        }
        _all_vars = {**_prob_vars, **_matrix_vars}
        _all_vars_dicts_ = [recurse_get_var_list_var(v) for v in _all_vars.values()]
        return {k: v for d in _all_vars_dicts_ for k, v in d.items()}

    def get_prob_vars(self) -> Mapping[str, _Variable]:
        _prob_vars = {
            k: v for p in self.get_problems() for k, v in p.get_prob_vars().items()
        }
        _matrix_vars = {
            k: v
            for m in self.get_matrices()
            for p in m.get_problems()
            for k, v in p.get_prob_vars().items()
        }
        # for g in self.get_systems():
        #     print(g)
        #     if isinstance(g, _SolverMatrix):
        #         for p in g.get_problems():
        #             print(p)
        #             for k, v in p.get_prob_vars().items():
        #                 print(k, v)
        #     else:
        #         for k, v in g.get_prob_vars().items():
        #             print(k, v)
        _all_vars = {**_prob_vars, **_matrix_vars}
        return _all_vars

    def get_systems(self) -> ValuesView[_Problem | _SolverMatrix]:
        return self.problems.values()

    def get_problems(self) -> Sequence[_Problem]:
        return [v for v in self.problems.values() if isinstance(v, _Problem)]

    def get_matrices(self) -> Sequence[_SolverMatrix]:
        return [v for v in self.problems.values() if isinstance(v, _SolverMatrix)]

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
    settings: dict[
        TolSettings | IterationSettings | Literal["CatchSolverErrors"],
        list[str | int | float | _Expression | _Variable],
    ] = dc.field(default_factory=dict)
    export_initial_condition: bool = True
    use_dynamic_topologies: bool | float = False
    _aux_vars: dict[str, _Variable] = dc.field(default_factory=dict)
    _deps_vars: dict[str, _Variable] = dc.field(default_factory=dict)

    def __repr__(self) -> str:
        return self.name

    def get_time_scheme(self) -> _TimeScheme:
        return self.time

    def get_aux_vars(self) -> ValuesView[_Variable]:
        _all_vars = {
            k: v for sg in self.sub_groups for k, v in sg.get_all_vars().items()
        }
        _dep_vars = {
            k: v for sg in self.sub_groups for k, v in sg.get_prob_vars().items()
        }
        check = all(item in _all_vars.items() for item in _dep_vars.items())
        if not check:
            raise ValueError(
                f"Dependent Variables not in super set check implementation"
            )
        _aux_vars = {k: v for k, v in _all_vars.items() if not k in _dep_vars}
        return _aux_vars.values()

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

    def AddAuxVariable(self, *var: _Variable):
        for v in var:
            if str(v) not in self._aux_vars:
                self._aux_vars[str(v)] = v

    def RemoveAuxVariable(self, *var: str | _Variable):
        for v in var:
            if isinstance(v, str):
                self._aux_vars.pop(v)
            else:
                self._aux_vars.pop(str(v))

    # SG
    def AddSolverSubGroup(self, *sg: _SolverSubGroup) -> None:
        for v in sg:
            # for x in v.get_aux_vars().values():
            #     self.AddAuxVariable(x)
            self.sub_groups.append(v)

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
        f.write(f"!DefSolverGroup={{{self.name}|{self.time}}}\n")
        # Handle Additional Vars
        vars = [str(v) for v in self.get_aux_vars()]
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
            _scale_res = (
                None
                if g.scale_first_residual is None
                else f"ScaleFirstResidual[{g.scale_first_residual}]"
            )
            f.write(
                f"!DefSolverSubGroup={{{join_fields(self,
                        g.get_method(), *g.get_systems(), _scale_res)}}}\n"
            )
            # if g.scale_first_residual:
            #     f.write(
            #         f'!DefSolverSubGroup={{{self.name}|{g.get_method()}|{
            #             "|".join(pobs)}|ScaleFirstResidual[{g.scale_first_residual}]}}\n'
            #     )
            # else:
            #     f.write(
            #         f"!DefSolverSubGroup={{{join_fields(self,
            #             g.get_method(), *pobs)}}}\n"
            #     )
