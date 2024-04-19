import enum
from typing import Literal, TextIO
from unittest.mock import patch

from ...pytools import get_enum, join_fields
from ...interface import *
from ..solid_mechanics.solid_problems import SolidProblem
from ...implementation.problems import BoundaryCondition


class L2SolidCalculationType(enum.StrEnum):
    cauchy_stress = "cauchy_stress"


L2_SOLID_CALCULATION_TYPE = Literal["cauchy_stress"]


class L2SolidProjection(_Problem):
    name: str
    solid_prob: SolidProblem
    calculation: L2SolidCalculationType = L2SolidCalculationType.cauchy_stress
    variables: dict[str, _Variable]
    bc: _BoundaryCondition
    aux_vars: dict[str, _Variable]
    problem: str = "l2solidprojection_problem"

    def __repr__(self) -> str:
        return self.name

    def get_variables(self) -> dict[str, _Variable]:
        return self.variables

    def get_aux_vars(self) -> dict[str, _Variable]:
        return self.aux_vars

    def get_bc_patches(self) -> list[_BCPatch]:
        patches = self.bc.get_patches()
        return [] if patches is None else patches

    def UseVariable(
        self, req: Literal["Space", "Variable"], var: _Variable
    ) -> None: ...

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
