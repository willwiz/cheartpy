import dataclasses as dc
from typing import Any, Literal, TextIO, TypedDict, overload
from cheartpy.cheart_core.aliases import SOLID_PROBLEM_TYPE, SolidProblemType
from cheartpy.cheart_core.pytools import get_enum, join_fields

from cheartpy.cheart_core.variables import Variable
from ..problems import BoundaryCondition, Problem
from .matlaws import Law, Matlaw


SOLID_VARIABLES = Literal[
    "Space",
    "Disp",
    "Velocity",
    "Pressure",
    "Fibers",
    "GenStruc",
]


SOLID_OPTIONS = Literal[
    "Density",
    "Perturbation-scale",
    "SetProblemTimeDiscretization",
]


class SolidProblem(Problem):
    name: str
    problem: SolidProblemType
    matlaws: list[Law]
    variables: dict[SOLID_VARIABLES, Variable]
    aux_vars: dict[str, Variable]
    options: dict[str, list[Any]]
    flags: list[str]
    bc: BoundaryCondition

    def __init__(self, name: str, problem: SolidProblemType, space: Variable, disp: Variable, vel: Variable | None = None, pres: Variable | None = None, matlaws: list[Law] | None = None) -> None:
        self.name = name
        self.problem = problem
        self.variables = {"Space": space, "Disp": disp}
        if problem is SolidProblemType.TRANSIENT and vel is None:
            raise ValueError(f"{name}: Transient problem must have velocity")
        if vel:
            self.variables["Velocity"] = vel
        if pres:
            self.variables["Pressure"] = pres
        self.matlaws = list() if matlaws is None else matlaws

    def AddMatlaw(self, *law: Matlaw):
        for v in law:
            self.matlaws.append(v)
            for k, x in v.aux_vars.items():
                self.aux_vars[k] = x

    def AddVariable(self, name: SOLID_VARIABLES, var: Variable) -> None:
        self.variables[name] = var

    @overload
    def UseOption(self, opt: Literal["Density"], val: float) -> None: ...

    @overload
    def UseOption(
        self, opt: Literal["Perturbation-scale"], val: float) -> None: ...

    @overload
    def UseOption(
        self, opt: Literal["SetProblemTimeDiscretization"], val: Literal["time_scheme_backward_euler"], sub_val: Literal["backward"]) -> None: ...

    def UseOption(self, opt: SOLID_OPTIONS, val: Any, *sub_val: Any) -> None:
        self.options[opt] = list(val, *sub_val)

    def write(self, f: TextIO):

        f.write(f"!DefProblem={{{self.name}|{self.problem}}}\n")
        for k, v in self.variables.items():
            f.write(f"  !UseVariablePointer={{{k}|{v.name}}}\n")
        for k, v in self.options.items():
            string = join_fields(v)
            f.write(f'  !{k}={{{string}}}\n')
        for v in self.flags:
            f.write(f"  !{v}\n")
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


@overload
def create_solid_problem(
    name: str, prob: Literal["TRANSIENT", SolidProblemType.TRANSIENT], space: Variable, disp: Variable, vel: Variable, pres: Variable | None = None
) -> SolidProblem: ...


@overload
def create_solid_problem(
    name: str, prob: Literal["QUASI_STATIC", SolidProblemType.QUASI_STATIC], space: Variable, disp: Variable, vel: Variable | None = None, pres: Variable | None = None
) -> SolidProblem: ...


def create_solid_problem(
    name: str, prob: SOLID_PROBLEM_TYPE | SolidProblemType, space: Variable, disp: Variable, vel: Variable | None = None, pres: Variable | None = None
) -> SolidProblem:
    problem = get_enum(prob, SolidProblemType)
    match problem, vel:
        case SolidProblemType.TRANSIENT, None:
            raise ValueError(f"Solid Problem {name}: Transient must have Vel")
        case _:
            pass
    return SolidProblem(name, problem, space, disp, vel, pres)
