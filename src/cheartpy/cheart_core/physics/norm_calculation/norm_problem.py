import dataclasses as dc
from typing import Literal, TextIO
from ...base_types.topologies import CheartTopology
from ...base_types.variables import Variable
from ...base_types.problems import BoundaryCondition, _Problem, BCPatch


class NormProblem(_Problem):
    name: str
    variables: dict[str, Variable]
    aux_vars: dict[str, Variable]
    bc: BoundaryCondition
    root_top: CheartTopology | None = None
    boundary_normal: int | None = None
    scale_by_measure: bool = False
    absolute_value: bool = False
    output_filename: str | None = None
    problem: str = "norm_calculation"

    def __init__(
        self,
        name: str,
        space: Variable,
        term1: Variable,
        term2: Variable | None = None,
        boundary_n: int | None = None,
    ) -> None:
        self.name = name
        self.variables = {"Space": space, "Term1": term1}
        if term2 is not None:
            self.variables["Term2"] = term2
        if boundary_n is not None:
            self.boundary_normal = boundary_n
        if term2 is not None != boundary_n is not None:
            raise ValueError(f"One of Term2 or Boundary normal must be None")
        self.aux_vars = dict()
        self.bc = BoundaryCondition()

    def __repr__(self) -> str:
        return self.name

    def get_variables(self) -> dict[str, Variable]:
        return self.variables

    def get_aux_vars(self) -> dict[str, Variable]:
        return self.aux_vars

    def get_bc_patches(self) -> list[BCPatch]:
        return [] if self.bc.patches is None else self.bc.patches

    def AddVariable(
        self, req: Literal["Space", "Term1", "Term2", "ExportToVariable"], var: Variable
    ) -> None:
        self.variables[req] = var

    def set_root_topology(self, top: CheartTopology) -> None:
        self.root_top = top

    def export_to_file(self, name: str) -> None:
        self.output_filename = name

    def write(self, f: TextIO):
        f.write(f"!DefProblem={{{self.name}|{self.problem}}}\n")
        for k, v in self.variables.items():
            f.write(f"  !UseVariablePointer={{{k}|{v.name}}}\n")
        if self.boundary_normal is not None:
            f.write(f"  !Boundary-normal={{{self.boundary_normal}}}\n")
        if self.scale_by_measure:
            f.write(f"  !scale-by-measure\n")
        if self.absolute_value:
            f.write(f"  !Absolute-value\n")
        if self.root_top is not None:
            f.write(f"  !SetRootTopology={{{repr(self.root_top)}}}\n")
        if self.output_filename is not None:
            f.write(f"  !Output-filename={{{self.output_filename}}}\n")
        self.bc.write(f)
