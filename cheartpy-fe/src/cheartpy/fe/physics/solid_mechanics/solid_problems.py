from typing import TYPE_CHECKING, Literal, TextIO, TypedDict, Unpack, overload

from cheartpy.fe.aliases import (
    SOLID_FLAGS,
    SOLID_OPTIONS,
    SOLID_PROBLEM_TYPE,
    SOLID_VARIABLES,
    SolidProblemType,
)
from cheartpy.fe.api import create_bc
from cheartpy.fe.string_tools import get_enum, join_fields
from cheartpy.fe.trait import (
    IBCPatch,
    IBoundaryCondition,
    IExpression,
    ILaw,
    IProblem,
    IVariable,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, ValuesView

__all__ = ["SolidProblem", "create_solid_mechanics_problem"]


class _SolidProblemExtraArgs(TypedDict, total=False):
    vel: IVariable | None
    pres: IVariable | None
    matlaws: list[ILaw] | None


class SolidProblem(IProblem):
    name: str
    problem: SolidProblemType
    matlaws: list[ILaw]
    variables: dict[SOLID_VARIABLES, IVariable]
    aux_vars: dict[str, IVariable]
    aux_expr: dict[str, IExpression]
    state_vars: dict[str, IVariable]
    options: dict[str, list[float | int | str]]
    gravity: tuple[float, tuple[float, float, float]] | IExpression | None
    flags: dict[str, bool]
    _buffering: bool
    bc: IBoundaryCondition

    def __repr__(self) -> str:
        return self.name

    def __init__(
        self,
        name: str,
        problem: SolidProblemType,
        space: IVariable,
        disp: IVariable,
        **kwargs: Unpack[_SolidProblemExtraArgs],
    ) -> None:
        vel = kwargs.get("vel")
        pres = kwargs.get("pres")
        matlaws = kwargs.get("matlaws")
        self.name = name
        self.problem = problem
        self.variables = {"Space": space, "Displacement": disp}
        if problem is SolidProblemType.TRANSIENT and vel is None:
            msg = f"{name}: Transient problem must have velocity"
            raise ValueError(msg)
        if vel:
            self.variables["Velocity"] = vel
        if pres:
            if pres.get_dim() != 1:
                msg = ">>>FATAL: Pressure variable for SolidProblems must have a dimension of 1"
                raise ValueError(msg)
            self.variables["Pressure"] = pres
        self.matlaws = [] if matlaws is None else matlaws
        self.aux_vars = {}
        self.aux_expr = {}
        self.state_vars = {}
        self.options = {}
        self.flags = {}
        self._buffering = True
        self.bc = create_bc()

    @property
    def buffering(self) -> bool:
        return self._buffering

    @buffering.setter
    def buffering(self, val: bool) -> None:
        self._buffering = val

    def get_prob_vars(self) -> Mapping[str, IVariable]:
        _self_vars_ = {str(v): v for v in self.variables.values()}
        return {**_self_vars_}

    def add_deps(self, *var: IVariable | IExpression | None) -> None:
        for v in var:
            if isinstance(v, IVariable):
                self.add_var_deps(v)
            else:
                self.add_expr_deps(v)

    def add_var_deps(self, *var: IVariable | None) -> None:
        for v in var:
            if v is None:
                continue
            if str(v) not in self.aux_vars:
                self.aux_vars[str(v)] = v

    def add_expr_deps(self, *expr: IExpression | None) -> None:
        for v in expr:
            if v is None:
                continue
            if str(v) not in self.aux_expr:
                self.aux_expr[str(v)] = v

    def get_var_deps(self) -> ValuesView[IVariable]:
        _vars_ = self.get_prob_vars()
        _w_vars_ = {str(v): v for w in self.matlaws for v in w.get_var_deps()}
        _b_vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        _s_vars_ = {str(v): v for v in self.state_vars.values()}
        _e_vars_ = {str(v): v for e in self.get_expr_deps() for v in e.get_var_deps()}
        return {**self.aux_vars, **_w_vars_, **_vars_, **_b_vars_, **_s_vars_, **_e_vars_}.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_ = {str(e): e for e in self.bc.get_expr_deps()}
        _w_exprs_ = {str(e): e for w in self.matlaws for e in w.get_expr_deps()}
        return {**_w_exprs_, **_expr_, **self.aux_expr}.values()

    def get_bc_patches(self) -> Sequence[IBCPatch]:
        patches = self.bc.get_patches()
        return [] if patches is None else list(patches)

    def add_matlaw(self, *law: ILaw) -> None:
        for w in law:
            self.matlaws.append(w)
            for v in w.get_var_deps():
                self.aux_vars[str(v)] = v

    def add_variable(self, name: SOLID_VARIABLES, var: IVariable) -> None:
        self.variables[name] = var

    def add_state_variable(self, *var: IVariable | IExpression | None) -> None:
        for v in var:
            if isinstance(v, IVariable):
                self.state_vars[str(v)] = v
                self.aux_vars[str(v)] = v

    def use_option(self, opt: SOLID_OPTIONS, val: str | float, *sub_val: str) -> None:
        self.options[opt] = [val, *sub_val]

    def stabilize(
        self,
        mode: Literal["UseStabilization", "Nearly-incompressible"],
        val: float,
        *order: int,
    ) -> None:
        self.options[mode] = [val, *order]

    @overload
    def add_gravity(self, g: float, direction: tuple[float, float, float]) -> None: ...
    @overload
    def add_gravity(self, g: IExpression) -> None: ...
    def add_gravity(
        self, g: float | IExpression, direction: tuple[float, float, float] | None = None
    ) -> None:
        match g, direction:
            case float(), tuple():
                self.gravity = (g, direction)
            case IExpression(), None:
                self.gravity = g
                self.aux_expr[str(g)] = g
            case _:
                msg = "Gravity must be either (float, tuple) or IExpression"
                raise ValueError(msg)

    def set_flags(self, flag: SOLID_FLAGS) -> None:
        self.flags[flag] = True

    def write(self, f: TextIO) -> None:
        f.write(f"!DefProblem={{{self.name}|{self.problem}}}\n")
        f.writelines(
            f"  !UseVariablePointer={{{join_fields(k, v)}}}\n" for k, v in self.variables.items()
        )
        if self.state_vars:
            f.write(
                f"  !Add-State-Variables={{{join_fields(*self.state_vars.values())}}}\n",
            )
        for k, v in self.options.items():
            string = join_fields(*v)
            f.write(f"  !{k}={{{string}}}\n")
        match self.gravity:
            case tuple():
                a, d = self.gravity
                f.write("  !Gravity-loading\n")
                f.write(f"    {'  '.join([str(x) for x in [*d, a]])}\n")
            case IExpression() as expr:
                f.write(f"  !Gravity-loading={{{expr}}}\n")
            case None:
                pass
        if not self._buffering:
            f.write("  !No-buffering\n")
        for k, v in self.flags.items():
            if v:
                f.write(f"  !{k}\n")
        for v in self.matlaws:
            f.write(v.string())
        self.bc.write(f)


def create_solid_mechanics_problem(
    name: str,
    prob: SOLID_PROBLEM_TYPE | SolidProblemType,
    space: IVariable,
    disp: IVariable,
    **kwargs: Unpack[_SolidProblemExtraArgs],
) -> SolidProblem:
    problem = get_enum(prob, SolidProblemType)
    if space.get_data() is None:
        msg = f"Space for {name} must be initialized with values"
        raise ValueError(msg)
    vel = kwargs.get("vel")
    match problem, vel:
        case SolidProblemType.TRANSIENT, None:
            msg = f"Solid Problem {name}: Transient must have Vel"
            raise ValueError(msg)
        case _:
            pass
    return SolidProblem(
        name,
        problem,
        space,
        disp,
        vel=vel,
        pres=kwargs.get("pres"),
        matlaws=kwargs.get("matlaws"),
    )
