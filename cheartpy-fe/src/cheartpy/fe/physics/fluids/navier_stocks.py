import dataclasses as dc
from typing import TYPE_CHECKING, Literal, Required, TextIO, TypedDict, Unpack, overload

from cheartpy.fe.impl import BoundaryCondition
from cheartpy.fe.trait import (
    IBCPatch,
    IBoundaryCondition,
    ICheartTopology,
    IExpression,
    IProblem,
    IVariable,
)
from cheartpy.fe.utils import join_fields

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, ValuesView

FluidProblemType = Literal["TRANSIENT_ALE_NONCON_NAVIER_STOKES_FLOW"]


TransientALENonConvNavierStokesFlowFlags = Literal["True-Navier-Poisson"]
TimeDiscretizationType = Literal["time_scheme_backward_euler"]
BackwordEulerTimeDiscSettings = Literal["backward"]


TransientALENonConvNavierStokesFlowVariables = Literal[
    "Space", "Velocity", "Pressure", "DomainVelocity"
]


class TransientALENonConvNavierStokesKwargs(TypedDict, total=False):
    space: Required[IVariable]
    vel: Required[IVariable]
    pres: Required[IVariable]
    dom_vel: Required[IVariable]
    viscosity: Required[float]
    density: float
    root_topology: ICheartTopology


_SettingValues = float | str | tuple[float | str, ...]


@dc.dataclass(slots=True, init=False)
class TransientALENonConvNavierStokesFlow(IProblem):
    name: str
    type: Literal["TRANSIENT_ALE_NONCON_NAVIER_STOKES_FLOW"]
    variables: dict[TransientALENonConvNavierStokesFlowVariables, IVariable]
    state_vars: dict[str, IVariable]
    root_topology: ICheartTopology
    flags: dict[str, bool]
    settings: dict[str, _SettingValues]
    bc: IBoundaryCondition
    buffering: bool
    time_discretization: tuple[str, ...]

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __init__(self, name: str, **kwargs: Unpack[TransientALENonConvNavierStokesKwargs]) -> None:
        vel = kwargs["vel"]
        self.name = name
        self.type = "TRANSIENT_ALE_NONCON_NAVIER_STOKES_FLOW"
        self.variables = {
            "Space": kwargs["space"],
            "Velocity": vel,
            "Pressure": kwargs["pres"],
            "DomainVelocity": kwargs["dom_vel"],
        }
        self.settings = {}
        self.settings["Viscosity"] = kwargs["viscosity"]
        self.settings["Density"] = kwargs.get("density", 1.0e-6)
        self.flags = {}
        self.root_topology = kwargs.get("root_topology") or vel.get_top()
        self.state_vars = {}
        self.bc = BoundaryCondition()
        self.time_discretization = ()
        self.buffering = True

    @overload
    def set_time_discretization(
        self,
        time_disc: Literal["time_scheme_backward_euler"],
        val: BackwordEulerTimeDiscSettings,
        /,
    ) -> None: ...
    @overload
    def set_time_discretization(self, time_disc: TimeDiscretizationType, *val: str) -> None: ...
    def set_time_discretization(self, time_disc: TimeDiscretizationType, *val: str) -> None:
        self.time_discretization = (time_disc, *val)

    def set_flag(self, flag: TransientALENonConvNavierStokesFlowFlags) -> None:
        self.flags[flag] = True

    def clear_flag(self, flag: TransientALENonConvNavierStokesFlowFlags) -> None:
        if flag in self.flags:
            self.flags[flag] = False

    def use_option(self, key: str, val: float | str) -> None:
        self.settings[key] = val

    def add_state_variable(self, *var: IVariable | IExpression | None) -> None:
        for v in var:
            if isinstance(v, IVariable):
                self.state_vars[str(v)] = v

    def get_prob_vars(self) -> Mapping[str, IVariable]:
        return {str(v): v for v in self.variables.values()}

    def add_deps(self, *var: IVariable | IExpression | None) -> None:
        for v in var:
            if isinstance(v, IVariable):
                self.add_var_deps(v)
            else:
                self.add_expr_deps(v)

    def add_var_deps(self, *var: IVariable | None) -> None: ...

    def add_expr_deps(self, *expr: IExpression | None) -> None: ...

    def get_var_deps(self) -> ValuesView[IVariable]:
        _vars_ = self.get_prob_vars()
        _b_vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        _s_vars_ = {str(v): v for v in self.state_vars.values()}
        _e_vars_ = {str(v): v for e in self.get_expr_deps() for v in e.get_var_deps()}
        return {**_vars_, **_b_vars_, **_s_vars_, **_e_vars_}.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_ = {str(e): e for e in self.bc.get_expr_deps()}
        return {**_expr_}.values()

    def get_bc_patches(self) -> Sequence[IBCPatch]:
        patches = self.bc.get_patches()
        return [] if patches is None else list(patches)

    def write(self, f: TextIO) -> None:
        _write_transient_navier_stokes(f, self)


ALEElementDependentStiffnessFlags = Literal["Radius-ratio-metric"]


ALEElementDependentStiffnessVariables = Literal[
    "Space", "DomainVelocity", "AleSpace", "ElementQuality", "ElementStiffness"
]


class ALEElementDependentStiffnessKwargs(TypedDict, total=False):
    space: Required[IVariable]
    dom_vel: Required[IVariable]
    ale_space: Required[IVariable]
    quality: Required[IVariable]
    stiffness: Required[IVariable]
    penalty: Required[float]
    root_topology: ICheartTopology


@dc.dataclass(slots=True, init=False)
class ALEElementDependentStiffness(IProblem):
    name: str
    type: Literal["ale_element_dependent_stiffness"]
    variables: dict[ALEElementDependentStiffnessVariables, IVariable]
    root_topology: ICheartTopology
    flags: dict[str, bool]
    settings: dict[str, _SettingValues]
    bc: IBoundaryCondition
    buffering: bool
    time_discretization: tuple[str, ...]
    state_vars: dict[str, IVariable]

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __init__(self, name: str, **kwargs: Unpack[ALEElementDependentStiffnessKwargs]) -> None:
        dom_vel = kwargs["dom_vel"]
        self.name = name
        self.type = "ale_element_dependent_stiffness"
        self.variables = {
            "Space": kwargs["space"],
            "DomainVelocity": dom_vel,
            "AleSpace": kwargs["ale_space"],
            "ElementQuality": kwargs["quality"],
            "ElementStiffness": kwargs["stiffness"],
        }
        self.settings = {}
        self.settings["Penalty-parameter"] = kwargs["penalty"]
        if density := kwargs.get("density"):
            self.settings["Density"] = density
        self.flags = {}
        self.root_topology = kwargs.get("root_topology") or dom_vel.get_top()
        self.buffering = True
        self.bc = BoundaryCondition()
        self.time_discretization = ()
        self.state_vars = {}

    @overload
    def set_time_discretization(
        self,
        time_disc: Literal["time_scheme_backward_euler"],
        val: BackwordEulerTimeDiscSettings,
        /,
    ) -> None: ...
    @overload
    def set_time_discretization(self, time_disc: TimeDiscretizationType, *val: str) -> None: ...
    def set_time_discretization(self, time_disc: TimeDiscretizationType, *val: str) -> None:
        self.time_discretization = (time_disc, *val)

    def set_flag(self, flag: ALEElementDependentStiffnessFlags) -> None:
        self.flags[flag] = True

    def clear_flag(self, flag: ALEElementDependentStiffnessFlags) -> None:
        if flag in self.flags:
            self.flags[flag] = False

    def use_option(self, key: str, val: float | str) -> None:
        self.settings[key] = val

    def add_state_variable(self, *var: IVariable | IExpression | None) -> None:
        for v in var:
            if isinstance(v, IVariable):
                self.state_vars[str(v)] = v

    def get_prob_vars(self) -> Mapping[str, IVariable]:
        return {str(v): v for v in self.variables.values()}

    def add_deps(self, *var: IVariable | IExpression | None) -> None:
        for v in var:
            if isinstance(v, IVariable):
                self.add_var_deps(v)
            else:
                self.add_expr_deps(v)

    def add_var_deps(self, *var: IVariable | None) -> None: ...

    def add_expr_deps(self, *expr: IExpression | None) -> None: ...

    def get_var_deps(self) -> ValuesView[IVariable]:
        _vars_ = self.get_prob_vars()
        _b_vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        _s_vars_ = {str(v): v for v in self.state_vars.values()}
        _e_vars_ = {str(v): v for e in self.get_expr_deps() for v in e.get_var_deps()}
        return {**_vars_, **_b_vars_, **_s_vars_, **_e_vars_}.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_ = {str(e): e for e in self.bc.get_expr_deps()}
        return {**_expr_}.values()

    def get_bc_patches(self) -> Sequence[IBCPatch]:
        patches = self.bc.get_patches()
        return [] if patches is None else list(patches)

    def write(self, f: TextIO) -> None:
        _write_transient_navier_stokes(f, self)


def _write_transient_navier_stokes(
    f: TextIO, problem: TransientALENonConvNavierStokesFlow | ALEElementDependentStiffness
) -> None:
    f.write(f"!DefProblem={{{problem.name}|{problem.type}}}\n")
    f.writelines(
        f"  !UseVariablePointer={{{join_fields(k, v)}}}\n" for k, v in problem.variables.items()
    )
    f.writelines(f"  !{k!s}\n" for k, v in problem.flags.items() if v)
    f.writelines(
        f"  !{k!s}={{{join_fields(*val) if isinstance(val, tuple) else val}}}\n"
        for k, val in problem.settings.items()
    )
    if problem.time_discretization:
        f.write(
            f"  !SetProblemTimeDiscretization={{{join_fields(*problem.time_discretization)}}}\n"
        )
    if problem.root_topology is not None:
        f.write(f"  !SetRootTopology={{{problem.root_topology}}}\n")
    if not problem.buffering:
        f.write("  !No-buffering\n")
    problem.bc.write(f)
