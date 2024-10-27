from typing import Literal, TextIO, ValuesView
from ...pytools import join_fields
from ...interface import *
from ...implementation import BoundaryCondition, CheartTopology


class NormProblem(_Problem):
    name: str
    variables: dict[str, _Variable]
    aux_vars: dict[str, _Variable]
    aux_expr: dict[str, _Expression]
    bc: _BoundaryCondition
    root_top: CheartTopology | None = None
    boundary_normal: int | None = None
    scale_by_measure: bool = False
    absolute_value: bool = False
    output_filename: str | None = None
    problem: str = "norm_calculation"

    def __init__(
        self,
        name: str,
        space: _Variable,
        term1: _Variable,
        term2: _Variable | None = None,
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
        self.aux_expr = dict()
        self.bc = BoundaryCondition()

    def __repr__(self) -> str:
        return self.name

    def get_variables(self) -> dict[str, _Variable]:
        return self.variables

    def add_aux_vars(self, *var: _Variable) -> None:
        for v in var:
            self.aux_vars[str(v)] = v

    def get_aux_vars(self) -> ValuesView[_Variable]:
        return self.aux_vars.values()

    def add_aux_expr(self, *expr: _Expression) -> None:
        for v in expr:
            self.aux_expr[str(v)] = v

    def get_bc_patches(self) -> list[_BCPatch]:
        patches = self.bc.get_patches()
        return [] if patches is None else patches

    def AddVariable(
        self,
        req: Literal["Space", "Term1", "Term2", "ExportToVariable"],
        var: _Variable,
    ) -> None:
        self.variables[req] = var

    def set_root_topology(self, top: CheartTopology) -> None:
        self.root_top = top

    def export_to_file(self, name: str) -> None:
        self.output_filename = name

    def write(self, f: TextIO):
        f.write(f"!DefProblem={{{join_fields(self, self.problem)}}}\n")
        for k, v in self.variables.items():
            f.write(f"  !UseVariablePointer={{{join_fields(k, v)}}}\n")
        if self.boundary_normal is not None:
            f.write(f"  !Boundary-normal={{{self.boundary_normal}}}\n")
        if self.scale_by_measure:
            f.write(f"  !scale-by-measure\n")
        if self.absolute_value:
            f.write(f"  !Absolute-value\n")
        if self.root_top is not None:
            f.write(f"  !SetRootTopology={{{str(self.root_top)}}}\n")
        if self.output_filename is not None:
            f.write(f"  !Output-filename={{{self.output_filename}}}\n")
        self.bc.write(f)
