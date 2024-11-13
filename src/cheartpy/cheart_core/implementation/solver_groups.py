__all__ = ["SolverSubGroup", "SolverGroup"]
import dataclasses as dc
from typing import Mapping, Sequence, TextIO, ValuesView
from ..aliases import *
from ..pytools import *
from ..interface import *
from .tools import recurse_get_var_list_expr, recurse_get_var_list_var

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
class SolverSubGroup(ISolverSubGroup):
    method: SolverSubgroupAlgorithm
    problems: dict[str, ISolverMatrix | IProblem] = dc.field(default_factory=dict)
    _scale_first_residual: float | None = None

    def get_method(self) -> SolverSubgroupAlgorithm:
        return self.method

    def get_all_vars(self) -> Mapping[str, IVariable]:
        _prob_vars = {str(v): v for p in self.get_problems() for v in p.get_var_deps()}
        _matrix_vars = {
            str(v): v
            for m in self.get_matrices()
            for p in m.get_problems()
            for v in p.get_var_deps()
        }
        _all_vars = {**_prob_vars, **_matrix_vars}
        _all_vars_dicts_ = [recurse_get_var_list_var(v) for v in _all_vars.values()]
        _prob_exprs = {
            str(e): e for p in self.get_problems() for e in p.get_expr_deps()
        }
        _matrix_exprs = {
            str(v): v
            for m in self.get_matrices()
            for p in m.get_problems()
            for v in p.get_expr_deps()
        }
        _all_exprs = {**_prob_exprs, **_matrix_exprs}
        _all_exprs_dicts_ = [recurse_get_var_list_expr(v) for v in _all_exprs.values()]
        return {
            k: v for d in _all_vars_dicts_ + _all_exprs_dicts_ for k, v in d.items()
        }

    def get_prob_vars(self) -> Mapping[str, IVariable]:
        _prob_vars = {
            k: v for p in self.get_problems() for k, v in p.get_prob_vars().items()
        }
        _matrix_vars = {
            k: v
            for m in self.get_matrices()
            for p in m.get_problems()
            for k, v in p.get_prob_vars().items()
        }
        _all_vars = {**_prob_vars, **_matrix_vars}
        return _all_vars

    def get_systems(self) -> ValuesView[IProblem | ISolverMatrix]:
        return self.problems.values()

    def get_problems(self) -> Sequence[IProblem]:
        return [v for v in self.problems.values() if isinstance(v, IProblem)]

    def get_matrices(self) -> Sequence[ISolverMatrix]:
        return [v for v in self.problems.values() if isinstance(v, ISolverMatrix)]

    @property
    def scale_first_residual(self) -> float | None:
        return self._scale_first_residual

    @scale_first_residual.setter
    def scale_first_residual(self, value: float | None) -> None:
        self._scale_first_residual = value


@dc.dataclass(slots=True)
class SolverGroup(ISolverGroup):
    name: str
    time: ITimeScheme
    sub_groups: list[ISolverSubGroup] = dc.field(default_factory=list)
    settings: dict[
        TolSettings | IterationSettings | Literal["CatchSolverErrors"],
        list[str | int | float | IExpression | IVariable],
    ] = dc.field(default_factory=dict)
    export_initial_condition: bool = True
    use_dynamic_topologies: bool | float = False
    _aux_vars: dict[str, IVariable] = dc.field(default_factory=dict)
    _deps_vars: dict[str, IVariable] = dc.field(default_factory=dict)

    def __repr__(self) -> str:
        return self.name

    def get_time_scheme(self) -> ITimeScheme:
        return self.time

    def get_aux_vars(self) -> ValuesView[IVariable]:
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

    def get_subgroups(self) -> Sequence[ISolverSubGroup]:
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

    def AddAuxVariable(self, *var: IVariable):
        for v in var:
            if str(v) not in self._aux_vars:
                self._aux_vars[str(v)] = v

    def RemoveAuxVariable(self, *var: str | IVariable):
        for v in var:
            if isinstance(v, str):
                self._aux_vars.pop(v)
            else:
                self._aux_vars.pop(str(v))

    # SG
    def AddSolverSubGroup(self, *sg: ISolverSubGroup) -> None:
        for v in sg:
            self.sub_groups.append(v)

    def RemoveSolverSubGroup(self, *sg: ISolverSubGroup) -> None:
        for v in sg:
            self.sub_groups.remove(v)

    def MakeSolverSubGroup(
        self,
        method: Literal["seq_fp_linesearch", "SOLVER_SEQUENTIAL"],
        *problems: ISolverMatrix | IProblem,
    ) -> None:
        self.sub_groups.append(
            SolverSubGroup(
                method=get_enum(method, SolverSubgroupAlgorithm),
                problems={str(p): p for p in problems},
            )
        )

    # WRITE
    def write(self, f: TextIO) -> None:
        f.write(hline("Solver Groups"))
        f.write(f"!DefSolverGroup={{{self}|{self.time}}}\n")
        # Handle Additional Vars
        vars = [str(v) for v in self.get_aux_vars()]
        for l in splicegen(45, vars):
            if l:
                f.write(
                    f'  !SetSolverGroup={{{join_fields(self, "AddVariables", *l)}}}\n'
                )
        # Print export init setting
        if self.export_initial_condition:
            f.write(f"  !SetSolverGroup={{{self}|export_initial_condition}}\n")
        # Print Conv Settings
        for k, v in self.settings.items():
            string = join_fields(self, k, *v)
            f.write(f"  !SetSolverGroup={{{string}}}\n")
        if self.use_dynamic_topologies:
            f.write(f"  !SetSolverGroup={{{self}|UsingDynamicTopologies}}\n")
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
