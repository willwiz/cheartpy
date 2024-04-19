import enum
from typing import Literal, TextIO

from ...pytools import get_enum

from ..solid_mechanics.solid_problems import SolidProblem
from ...base_types.variables import Variable
from ...base_types.problems import BoundaryCondition, _Problem, BCPatch


class L2SolidCalculationType(enum.StrEnum):
    cauchy_stress = "cauchy_stress"


L2_SOLID_CALCULATION_TYPE = Literal["cauchy_stress"]


class L2SolidProjection(_Problem):
    name: str
    solid_prob: SolidProblem
    calculation: L2SolidCalculationType = L2SolidCalculationType.cauchy_stress
    variables: dict[str, Variable]
    bc: BoundaryCondition
    aux_vars: dict[str, Variable]
    problem: str = "l2solidprojection_problem"

    def __repr__(self) -> str:
        return self.name

    def get_variables(self) -> dict[str, Variable]:
        return self.variables

    def get_aux_vars(self) -> dict[str, Variable]:
        return self.aux_vars

    def get_bc_patches(self) -> list[BCPatch]:
        return [] if self.bc.patches is None else self.bc.patches

    def UseVariable(self, req: Literal["Space", "Variable"], var: Variable) -> None: ...

    def __init__(
        self,
        name: str,
        space: Variable,
        var: Variable,
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
            f.write(f"  !UseVariablePointer={{{k}|{v.name}}}\n")

        f.write(f"  !Mechanical-Problem={{{self.solid_prob}}}\n")
        f.write(f"  !Projected-Variable={{{self.calculation}}}\n")
        self.bc.write(f)
