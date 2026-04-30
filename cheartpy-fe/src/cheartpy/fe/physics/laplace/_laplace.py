import dataclasses as dc
from typing import TYPE_CHECKING, TextIO

from cheartpy.fe.api import create_bc
from cheartpy.fe.trait import IBCPatch, IBoundaryCondition, IExpression, IProblem, IVariable

if TYPE_CHECKING:
    from collections.abc import Sequence, ValuesView


@dc.dataclass(slots=True, init=False)
class LaplaceProblem(IProblem):
    name: str
    variables: dict[str, IVariable]
    state_vars: dict[str, IVariable]
    var_deps: dict[str, IVariable]
    expr_deps: dict[str, IExpression]
    bc: IBoundaryCondition
    buffering: bool = False

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __init__(
        self,
        name: str,
        space: IVariable,
        var: IVariable,
    ) -> None:
        self.name = name
        self.variables = {"Space": space, "Variable": var}
        self.aux_vars = {}
        self.aux_expr = {}
        self.state_vars = {}
        self.bc = create_bc()

    def add_state_variable(self, *var: IVariable | IExpression | None) -> None:
        for v in var:
            if isinstance(v, IVariable):
                self.state_vars[str(v)] = v

    def get_prob_vars(self) -> dict[str, IVariable]:
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
            if str(v) not in self.var_deps:
                self.var_deps[str(v)] = v

    def add_expr_deps(self, *expr: IExpression | None) -> None:
        for e in expr:
            if e is None:
                continue
            if str(e) not in self.expr_deps:
                self.expr_deps[str(e)] = e

    def get_var_deps(self) -> ValuesView[IVariable]:
        _vars_ = self.get_prob_vars()
        _b_vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        _e_vars_ = {str(v): v for e in self.get_expr_deps() for v in e.get_var_deps()}
        return {**self.var_deps, **_vars_, **_b_vars_, **_e_vars_}.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_ = {str(e): e for e in self.bc.get_expr_deps()}
        return {**_expr_, **self.expr_deps}.values()

    def get_bc_patches(self) -> Sequence[IBCPatch]:
        patchs = self.bc.get_patches() or []
        return list(patchs)

    def write(self, f: TextIO) -> None:
        _write_laplace(self, f)


def _write_laplace(p: LaplaceProblem, f: TextIO) -> None:
    f.write(f"!DefProblem={{{p.name}|LAPLACE_PROBLEM}}")
    f.writelines(f"  !UseVariablePointer={{{k}|{v}}}\n" for k, v in p.variables.items())
    p.bc.write(f)
