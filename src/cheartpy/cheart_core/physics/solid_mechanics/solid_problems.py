from typing import Any, Literal, Mapping, Sequence, TextIO, ValuesView
from ...aliases import SOLID_PROBLEM_TYPE, SolidProblemType
from ...pytools import get_enum, join_fields
from ...implementation.problems import BoundaryCondition
from ...interface import *
from .matlaws import _Law


SOLID_VARIABLES = Literal[
    "Space",
    "Disp",
    "Velocity",
    "Pressure",
    "Fibers",
    "GenStruc",
]


SOLID_OPTIONS = Literal[
    "Density", "Perturbation-scale", "SetProblemTimeDiscretization", "UseStabilization"
]

SOLID_FLAGS = Literal["Inverse-mechanics",]


class SolidProblem(_Problem):
    name: str
    problem: SolidProblemType
    matlaws: list[_Law]
    variables: dict[str, _Variable]
    aux_vars: dict[str, _Variable]
    aux_expr: dict[str, _Expression]
    state_vars: dict[str, _Variable]
    options: dict[str, list[Any]]
    flags: dict[str, bool]
    bc: _BoundaryCondition

    def __repr__(self) -> str:
        return self.name

    def get_prob_vars(self) -> Mapping[str, _Variable]:
        _self_vars_ = {str(v): v for v in self.variables.values()}
        _vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        return {**_self_vars_, **_vars_}

    def add_var_deps(self, *var: _Variable) -> None:
        for v in var:
            if str(v) not in self.aux_vars:
                self.aux_vars[str(v)] = v

    def add_expr_deps(self, *expr: _Expression) -> None:
        for v in expr:
            if str(v) not in self.aux_expr:
                self.aux_expr[str(v)] = v

    def get_var_deps(self) -> ValuesView[_Variable]:
        _vars_ = self.get_prob_vars()
        _w_vars_ = {str(v): v for w in self.matlaws for v in w.get_var_deps()}
        return {**_vars_, **_w_vars_, **self.aux_vars}.values()

    def get_expr_deps(self) -> ValuesView[_Expression]:
        _expr_ = {str(e): e for e in self.bc.get_expr_deps()}
        _w_exprs_ = {str(e): e for w in self.matlaws for e in w.get_expr_deps()}
        return {**_expr_, **_w_exprs_, **self.aux_expr}.values()

    def get_bc_patches(self) -> Sequence[_BCPatch]:
        patches = self.bc.get_patches()
        return list() if patches is None else patches

    def __init__(
        self,
        name: str,
        problem: SolidProblemType,
        space: _Variable,
        disp: _Variable,
        vel: _Variable | None = None,
        pres: _Variable | None = None,
        matlaws: list[_Law] | None = None,
    ) -> None:
        self.name = name
        self.problem = problem
        self.variables = {"Space": space, "Displacement": disp}
        if problem is SolidProblemType.TRANSIENT and vel is None:
            raise ValueError(f"{name}: Transient problem must have velocity")
        if vel:
            self.variables["Velocity"] = vel
        if pres:
            if pres.get_dim() != 1:
                raise ValueError(
                    ">>>FATAL: Pressure variable for SolidProblems must have a dimension of 1"
                )
            self.variables["Pressure"] = pres
        self.matlaws = list() if matlaws is None else matlaws
        self.aux_vars = dict()
        self.aux_expr = dict()
        self.state_vars = dict()
        self.options = dict()
        self.flags = dict()
        self.bc = BoundaryCondition()

    def AddMatlaw(self, *law: _Law):
        for w in law:
            self.matlaws.append(w)
            for v in w.get_var_deps():
                self.aux_vars[str(v)] = v

    def AddVariable(self, name: SOLID_VARIABLES, var: _Variable) -> None:
        self.variables[name] = var

    def AddStateVariable(self, *var: _Variable) -> None:
        for v in var:
            self.state_vars[str(v)] = v
            self.aux_vars[str(v)] = v

    def UseOption(self, opt: SOLID_OPTIONS, val: Any, *sub_val: Any) -> None:
        self.options[opt] = list([val, *sub_val])

    def UseStabilization(self, val: float, order: int) -> None:
        self.options["UseStabilization"] = [f"{val} {order}"]

    def SetFlags(self, flag: SOLID_FLAGS) -> None:
        self.flags[flag] = True

    def write(self, f: TextIO):
        f.write(f"!DefProblem={{{self.name}|{self.problem}}}\n")
        for k, v in self.variables.items():
            f.write(f"  !UseVariablePointer={{{join_fields(k, v)}}}\n")
        if self.state_vars:
            f.write(
                f"  !Add-State-Variables={{{join_fields(*self.state_vars.values())}}}\n"
            )
        for k, v in self.options.items():
            if k == "UseStabilization":
                f.write(f"  !{k}\n")
                f.write(f"    {" ".join(v)}\n")
            else:
                string = join_fields(*v)
                f.write(f"  !{k}={{{string}}}\n")
        for k, v in self.flags.items():
            if v:
                f.write(f"  !{k}\n")
        for v in self.matlaws:
            f.write(v.string())
        self.bc.write(f)

    # vars: dict[str, Variable] = dc.field(default_factory=dict)
    # aux_vars: dict[str, Variable] = dc.field(default_factory=dict)
    # options: dict[str, list[str]] = dc.field(default_factory=dict)
    # flags: list[str] = dc.field(default_factory=list)
    # BC: BoundaryCondition = dc.field(default_factory=BoundaryCondition)

    # def UseVariable(self, req: str, var: Variable) -> None:
    #     self.vars[req] = var

    # def UseOption(self, opt: str, *val: str) -> None:
    #     if val:
    #         self.options[opt] = list(val)
    #     else:
    #         self.flags.append(opt)


def create_solid_mechanics_problem(
    name: str,
    prob: SOLID_PROBLEM_TYPE | SolidProblemType,
    space: _Variable,
    disp: _Variable,
    vel: _Variable | None = None,
    pres: _Variable | None = None,
) -> SolidProblem:
    problem = get_enum(prob, SolidProblemType)
    if space.get_data() is None:
        raise ValueError(f"Space for {name} must be initialized with values")
    match problem, vel:
        case SolidProblemType.TRANSIENT, None:
            raise ValueError(f"Solid Problem {name}: Transient must have Vel")
        case _:
            pass
    return SolidProblem(name, problem, space, disp, vel, pres)
