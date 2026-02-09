from typing import TYPE_CHECKING, Literal, TextIO

from cheartpy.fe.api import create_bc
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

__all__ = ["NormProblem"]


class NormProblem(IProblem):
    name: str
    variables: dict[str, IVariable | IExpression]
    aux_vars: dict[str, IVariable]
    aux_expr: dict[str, IExpression]
    bc: IBoundaryCondition
    root_top: ICheartTopology | None = None
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
        term1: IExpression | IVariable,
        term2: IExpression | IVariable | None = None,
        boundary_n: int | None = None,
    ) -> None:
        self.name = name
        self.variables = {"Space": space, "Term1": term1}
        if term2 is not None:
            self.variables["Term2"] = term2
        self.boundary_normal = boundary_n
        if term2 is not None != boundary_n is not None:
            msg = "One of Term2 or Boundary normal must be None"
            raise ValueError(msg)
        self.aux_vars = {}
        self.aux_expr = {}
        self.bc = create_bc()
        self.root_top = None
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
        _self_vars_ = {str(v): v for v in self.variables.values() if isinstance(v, IVariable)}
        for v in self.variables.values():
            if isinstance(v, IExpression):
                for dep in v.get_var_deps():
                    _self_vars_[str(dep)] = dep
        # _vars_ = {str(v): v for v in self.bc.get_vars_deps()}
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
        _b_vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        return {**_vars_, **_b_vars_, **self.aux_vars}.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_ = {str(e): e for e in self.variables.values() if isinstance(e, IExpression)}
        _b_expr_ = {str(e): e for e in self.bc.get_expr_deps()}
        return {**_expr_, **_b_expr_, **self.aux_expr}.values()

    def get_bc_patches(self) -> Sequence[IBCPatch]:
        patches = self.bc.get_patches()
        return [] if patches is None else list(patches)

    def add_state_variable(self, *var: IVariable | IExpression | None) -> None:
        pass

    def add_variable(
        self,
        req: Literal["Space", "Term1", "Term2", "ExportToVariable"],
        var: IVariable,
    ) -> None:
        self.variables[req] = var

    def set_root_topology(self, top: ICheartTopology) -> None:
        self.root_top = top

    def export_to_file(self, name: str) -> None:
        self.output_filename = name

    def write(self, f: TextIO) -> None:
        f.write(f"!DefProblem={{{join_fields(self, self._problem_name)}}}\n")
        f.writelines(
            f"  !UseVariablePointer={{{join_fields(k, v)}}}\n" for k, v in self.variables.items()
        )
        if self.boundary_normal is not None:
            f.write(f"  !Boundary-normal={{{self.boundary_normal}}}\n")
        if self.scale_by_measure:
            f.write("  !scale-by-measure\n")
        if self.absolute_value:
            f.write("  !Absolute-value\n")
        if self.root_top is not None:
            f.write(f"  !SetRootTopology={{{self.root_top!s}}}\n")
        if not self._buffering:
            f.write("  !No-buffering\n")
        if self.output_filename is not None:
            f.write(f"  !Output-filename={{{self.output_filename}}}\n")
        self.bc.write(f)
