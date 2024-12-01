__all__ = ["NormProblem"]
from typing import Literal, Mapping, Sequence, TextIO, ValuesView

from ...trait.basic import IExpression, IVariable
from ...pytools import join_fields
from ...trait import *
from ...impl import BoundaryCondition, CheartTopology


class NormProblem(IProblem):
    name: str
    variables: dict[str, IVariable]
    aux_vars: dict[str, IVariable]
    aux_expr: dict[str, IExpression]
    bc: IBoundaryCondition
    root_top: CheartTopology | None = None
    boundary_normal: int | None = None
    scale_by_measure: bool = False
    absolute_value: bool = False
    output_filename: str | None = None
    _buffering: bool = False
    _problem_name: str = "norm_calculation"

    def __repr__(self) -> str:
        return self.name

    def __init__(
        self,
        name: str,
        space: IVariable,
        term1: IVariable,
        term2: IVariable | None = None,
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
        self.root_top = None
        self.boundary_normal = None
        self.scale_by_measure = False
        self.absolute_value = False
        self.output_filename = None
        self._buffering = True
        self._problem_name = "norm_calculation"

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

    def add_deps(self, *vars: IVariable | IExpression) -> None:
        for v in vars:
            if isinstance(v, IVariable):
                self.add_var_deps(v)
            else:
                self.add_expr_deps(v)

    def add_var_deps(self, *var: IVariable) -> None:
        for v in var:
            self.aux_vars[str(v)] = v

    def add_expr_deps(self, *expr: IExpression) -> None:
        for v in expr:
            self.aux_expr[str(v)] = v

    def get_var_deps(self) -> ValuesView[IVariable]:
        _vars_ = self.get_prob_vars()
        _b_vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        return {**_vars_, **_b_vars_, **self.aux_vars}.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_ = {str(e): e for e in self.bc.get_expr_deps()}
        return {**_expr_, **self.aux_expr}.values()

    def get_bc_patches(self) -> Sequence[IBCPatch]:
        patches = self.bc.get_patches()
        return list() if patches is None else list(patches)

    def AddVariable(
        self,
        req: Literal["Space", "Term1", "Term2", "ExportToVariable"],
        var: IVariable,
    ) -> None:
        self.variables[req] = var

    def set_root_topology(self, top: CheartTopology) -> None:
        self.root_top = top

    def export_to_file(self, name: str) -> None:
        self.output_filename = name

    def write(self, f: TextIO):
        f.write(f"!DefProblem={{{join_fields(self, self._problem_name)}}}\n")
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
        if self._buffering == False:
            f.write(f"  !No-buffering\n")
        if self.output_filename is not None:
            f.write(f"  !Output-filename={{{self.output_filename}}}\n")
        self.bc.write(f)
