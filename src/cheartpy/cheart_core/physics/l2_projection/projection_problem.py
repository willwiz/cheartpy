import enum
from typing import Literal, Mapping, Sequence, TextIO, ValuesView
from ...pytools import get_enum, join_fields
from ...interface import *
from ...implementation import BoundaryCondition
from ..solid_mechanics.solid_problems import SolidProblem


class L2SolidCalculationType(enum.StrEnum):
    cauchy_stress = "cauchy_stress"
    deformation_gradient = "deformation_gradient"


L2_SOLID_CALCULATION_TYPE = Literal["cauchy_stress", "deformation_gradient"]


class L2SolidProjection(_Problem):
    name: str
    solid_prob: SolidProblem
    calculation: L2SolidCalculationType = L2SolidCalculationType.cauchy_stress
    variables: dict[str, _Variable]
    bc: _BoundaryCondition
    aux_vars: dict[str, _Variable]
    aux_expr: dict[str, _Expression]
    problem: str = "l2solidprojection_problem"

    def __repr__(self) -> str:
        return self.name

    def get_prob_vars(self) -> Mapping[str, _Variable]:
        _self_vars_ = {str(v): v for v in self.variables.values()}
        _vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        return {**_self_vars_, **_vars_}

    def add_var_deps(self, *var: _Variable) -> None:
        for v in var:
            self.aux_vars[str(v)] = v

    def add_expr_deps(self, *expr: _Expression) -> None:
        for v in expr:
            self.aux_expr[str(v)] = v

    def get_var_deps(self) -> ValuesView[_Variable]:
        _vars_ = self.get_prob_vars()
        return {**_vars_, **self.aux_vars}.values()

    def get_expr_deps(self) -> ValuesView[_Expression]:
        _expr_ = {str(e): e for e in self.bc.get_expr_deps()}
        return {**_expr_, **self.aux_expr}.values()

    def get_bc_patches(self) -> Sequence[_BCPatch]:
        patches = self.bc.get_patches()
        return list() if patches is None else patches

    def set_projection(
        self, calc: L2SolidCalculationType | L2_SOLID_CALCULATION_TYPE
    ) -> None:
        self.calculation = get_enum(calc, L2SolidCalculationType)

    def __init__(
        self,
        name: str,
        space: _Variable,
        var: _Variable,
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
        self.bc = BoundaryCondition()

    def write(self, f: TextIO):
        f.write(f"!DefProblem={{{self.name}|{self.problem}}}\n")
        for k, v in self.variables.items():
            f.write(f"  !UseVariablePointer={{{join_fields(k, v)}}}\n")

        f.write(f"  !Mechanical-Problem={{{self.solid_prob}}}\n")
        f.write(f"  !Projected-Variable={{{self.calculation}}}\n")
        self.bc.write(f)
