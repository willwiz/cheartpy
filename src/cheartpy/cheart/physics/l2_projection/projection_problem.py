from __future__ import annotations

__all__ = ["L2SolidProjection"]
import enum
from collections.abc import Mapping, Sequence, ValuesView
from typing import Literal, TextIO

from ...api import create_bc
from ...pytools import get_enum, join_fields
from ...trait import *
from ..solid_mechanics.solid_problems import SolidProblem


class L2SolidCalculationType(enum.StrEnum):
    cauchy_stress = "cauchy_stress"
    deformation_gradient = "deformation_gradient"


L2_SOLID_CALCULATION_TYPE = Literal["cauchy_stress", "deformation_gradient"]


class L2SolidProjection(IProblem):
    name: str
    solid_prob: SolidProblem
    calculation: L2SolidCalculationType = L2SolidCalculationType.cauchy_stress
    variables: dict[str, IVariable]
    aux_vars: dict[str, IVariable]
    aux_expr: dict[str, IExpression]
    bc: IBoundaryCondition
    _buffering: bool = False
    _problem: str = "l2solidprojection_problem"

    def __repr__(self) -> str:
        return self.name

    def __init__(
        self,
        name: str,
        space: IVariable,
        var: IVariable,
        solid_prob: SolidProblem,
        projected_var: (
            L2SolidCalculationType | L2_SOLID_CALCULATION_TYPE
        ) = L2SolidCalculationType.cauchy_stress,
    ) -> None:
        self.name = name
        self.solid_prob = solid_prob
        self.variables = {"Space": space, "Variable": var}
        self.calculation = get_enum(projected_var, L2SolidCalculationType)
        self.aux_vars = dict()
        self.aux_expr = dict()
        self.bc = create_bc()
        self._buffering = True

    @property
    def buffering(self) -> bool:
        return self._buffering

    @buffering.setter
    def buffering(self, val: bool) -> None:
        self._buffering = val

    def get_prob_vars(self) -> Mapping[str, IVariable]:
        _self_vars_ = {str(v): v for v in self.variables.values()}
        # _vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        return {**_self_vars_}

    def add_deps(self, *vars: IVariable | IExpression | None) -> None:
        for v in vars:
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
        _b_vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        return {**_vars_, **_b_vars_, **self.aux_vars}.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_ = {str(e): e for e in self.bc.get_expr_deps()}
        return {**_expr_, **self.aux_expr}.values()

    def add_state_variable(self, *var: IVariable | IExpression | None) -> None:
        return

    def get_bc_patches(self) -> Sequence[IBCPatch]:
        patches = self.bc.get_patches()
        return list() if patches is None else list(patches)

    def set_projection(
        self,
        calc: L2SolidCalculationType | L2_SOLID_CALCULATION_TYPE,
    ) -> None:
        self.calculation = get_enum(calc, L2SolidCalculationType)

    def write(self, f: TextIO):
        f.write(f"!DefProblem={{{self.name}|{self._problem}}}\n")
        f.writelines(
            f"  !UseVariablePointer={{{join_fields(k, v)}}}\n" for k, v in self.variables.items()
        )

        f.write(f"  !Mechanical-Problem={{{self.solid_prob}}}\n")
        f.write(f"  !Projected-Variable={{{self.calculation}}}\n")
        if self._buffering == False:
            f.write("  !No-buffering\n")
        self.bc.write(f)
