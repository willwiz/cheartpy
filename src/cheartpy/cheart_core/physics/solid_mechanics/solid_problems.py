import dataclasses as dc
from typing import Any, Literal, TextIO, TypedDict, overload
from cheartpy.cheart_core.aliases import SOLID_PROBLEM_TYPE, SolidProblemType
from cheartpy.cheart_core.pytools import get_enum, join_fields
from ...implementation.problems import BoundaryCondition
from ...interface import *
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


class SolidProblem(_Problem):
    name: str
    problem: SolidProblemType
    matlaws: list[Law]
    variables: dict[str, _Variable]
    aux_vars: dict[str, _Variable]
    options: dict[str, list[Any]]
    flags: dict[str, None]
    bc: BoundaryCondition

    def __repr__(self) -> str:
        return self.name

    def get_variables(self) -> dict[str, _Variable]:
        return self.variables

    def get_aux_vars(self) -> dict[str, _Variable]:
        return self.aux_vars

    def get_bc_patches(self) -> list[_BCPatch]:
        return [] if self.bc.patches is None else self.bc.patches

    def __init__(
        self,
        name: str,
        problem: SolidProblemType,
        space: _Variable,
        disp: _Variable,
        vel: _Variable | None = None,
        pres: _Variable | None = None,
        matlaws: list[Law] | None = None,
    ) -> None:
        self.name = name
        self.problem = problem
        self.variables = {"Space": space, "Displacement": disp}
        if problem is SolidProblemType.TRANSIENT and vel is None:
            raise ValueError(f"{name}: Transient problem must have velocity")
        if vel:
            self.variables["Velocity"] = vel
        if pres:
            self.variables["Pressure"] = pres
        self.matlaws = list() if matlaws is None else matlaws
        self.aux_vars = dict()
        self.options = dict()
        self.flags = dict()
        self.bc = BoundaryCondition()

    def AddMatlaw(self, *law: Law):
        for v in law:
            self.matlaws.append(v)
            for k, x in v.get_aux_vars().items():
                self.aux_vars[k] = x

    def AddVariable(self, name: SOLID_VARIABLES, var: _Variable) -> None:
        self.variables[name] = var

    def UseOption(self, opt: SOLID_OPTIONS, val: Any, *sub_val: Any) -> None:
        self.options[opt] = list([val, *sub_val])

    def write(self, f: TextIO):
        f.write(f"!DefProblem={{{self.name}|{self.problem}}}\n")
        for k, v in self.variables.items():
            f.write(f"  !UseVariablePointer={{{join_fields(k, v)}}}\n")
        for k, v in self.options.items():
            string = join_fields(*v)
            f.write(f"  !{k}={{{string}}}\n")
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


def create_solid_problem(
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
