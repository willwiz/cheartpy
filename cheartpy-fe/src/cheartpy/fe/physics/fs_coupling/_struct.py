import dataclasses as dc
from typing import TYPE_CHECKING, Literal, TextIO

from cheartpy.fe.api import create_bc
from cheartpy.fe.string_tools import join_fields
from cheartpy.fe.trait import (
    IBCPatch,
    IBoundaryCondition,
    ICheartTopology,
    IExpression,
    IProblem,
    IVariable,
)

if TYPE_CHECKING:
    from collections.abc import ValuesView

__all__ = ["FSCouplingProblem", "FSExpr"]


@dc.dataclass(slots=True)
class FSExpr:
    var: IVariable | IExpression
    mult: IExpression | IVariable | float | None = None
    op: Literal["trace", "dt"] | None = None

    def to_str(self) -> str:
        mult = join_fields(self.mult, self.op, char=";")
        if mult == "":
            mult = "1"
        return f"{self.var}[{mult}]"

    def get_var_deps(self) -> ValuesView[IVariable]:
        _vars_: dict[str, IVariable] = {}
        if isinstance(self.var, IVariable):
            _vars_[str(self.var)] = self.var
        if isinstance(self.var, IExpression):
            for v in self.var.get_var_deps():
                if str(v) not in _vars_:
                    _vars_[str(v)] = v
        if isinstance(self.mult, IVariable) and str(self.mult) not in _vars_:
            _vars_[str(self.mult)] = self.mult
        if isinstance(self.mult, IExpression):
            for v in self.mult.get_var_deps():
                if str(v) not in _vars_:
                    _vars_[str(v)] = v
        return _vars_.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_: dict[str, IExpression] = {}
        if isinstance(self.var, IExpression):
            _expr_[str(self.var)] = self.var
            for e in self.var.get_expr_deps():
                _expr_[str(e)] = e
        if isinstance(self.var, IVariable):
            for e in self.var.get_expr_deps():
                _expr_[str(e)] = e
        if isinstance(self.mult, IExpression):
            _expr_[str(self.mult)] = self.mult
            for e in self.mult.get_expr_deps():
                _expr_[str(e)] = e
        if isinstance(self.mult, IVariable):
            for e in self.mult.get_expr_deps():
                _expr_[str(e)] = e
        return _expr_.values()


@dc.dataclass(slots=True)
class FSCouplingTerm:
    test_var: IVariable
    terms: list[FSExpr]

    def get_var_deps(self) -> ValuesView[IVariable]:
        _vars_: dict[str, IVariable] = {str(self.test_var): self.test_var}
        for t in self.terms:
            for v in t.get_var_deps():
                _vars_[str(v)] = v
        return _vars_.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_: dict[str, IExpression] = {}
        for t in self.terms:
            for e in t.get_expr_deps():
                _expr_[str(e)] = e
        return _expr_.values()


class FSCouplingProblem(IProblem):
    name: str
    space: IVariable
    root_topology: ICheartTopology | None
    lm: FSCouplingTerm | None
    m_terms: dict[str, FSCouplingTerm]
    aux_vars: dict[str, IVariable]
    aux_expr: dict[str, IExpression]
    bc: IBoundaryCondition
    perturbation: bool = True
    _buffering: bool = True
    _problem_name: str = "fscoupling_problem"

    def __repr__(self) -> str:
        return self.name

    def __init__(
        self,
        name: str,
        space: IVariable,
        root_top: ICheartTopology | None = None,
    ) -> None:
        self.name = name
        self.space = space
        self.root_topology = root_top if root_top else None
        self.lm = None
        self.m_terms = {}
        self.aux_vars = {}
        self.aux_expr = {}
        self.bc = create_bc()
        self._buffering = True

    @property
    def buffering(self) -> bool:
        return self._buffering

    @buffering.setter
    def buffering(self, val: bool) -> None:
        self._buffering = val

    def set_lagrange_mult(self, var: IVariable, *expr: FSExpr) -> None:
        self.lm = FSCouplingTerm(var, list(expr))

    def add_term(self, var: IVariable, *expr: FSExpr) -> None:
        self.m_terms[str(var)] = FSCouplingTerm(var, list(expr))

    def add_state_variable(self, *var: IVariable | IExpression | None) -> None:
        for v in var:
            if v is None:
                continue
            if not isinstance(v, IVariable):
                continue
            if str(v) not in self.m_terms:
                self.m_terms[str(v)] = FSCouplingTerm(v, [FSExpr(v, 0)])

    def get_prob_vars(self) -> dict[str, IVariable]:
        variables: dict[str, IVariable] = {str(self.space): self.space}
        if self.lm is not None:
            variables[str(self.lm.test_var)] = self.lm.test_var
        for v in self.m_terms.values():
            if str(v.test_var) not in variables:
                variables[str(v.test_var)] = v.test_var
        return variables

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
        _vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        _vars_[str(self.space)] = self.space
        if self.lm is not None:
            for t in self.lm.get_var_deps():
                _vars_[str(t)] = t
        for v in self.m_terms.values():
            for t_var in v.get_var_deps():
                _vars_[str(t_var)] = t_var
        return {**self.aux_vars, **_vars_}.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_ = {str(e): e for e in self.bc.get_expr_deps()}
        if self.lm is not None:
            for t in self.lm.get_expr_deps():
                _expr_[str(t)] = t
        for t in self.m_terms.values():
            for e in t.get_expr_deps():
                _expr_[str(e)] = e
        return {**self.aux_expr, **_expr_}.values()

    def get_bc_patches(self) -> list[IBCPatch]:
        patches = self.bc.get_patches()
        return [] if patches is None else list(patches)

    def write(self, f: TextIO) -> None:
        f.write(f"!DefProblem={{{self}|{self._problem_name}}}\n")
        f.write(f"  !UseVariablePointer={{Space|{self.space}}}\n")
        f.writelines(
            (
                f"  !Addterms={{TestVariable[{t.test_var}]|"
                f"{' '.join([s.to_str() for s in t.terms])}}}\n"
            )
            for t in self.m_terms.values()
        )
        if self.lm is not None:
            f.write(
                f"  !Addterms={{TestVariable[{self.lm.test_var}*]|"
                f"{' '.join([s.to_str() for s in self.lm.terms])}}}\n",
            )
        else:
            msg = "Lagrange multiplier not set for FSCouplingProblem"
            raise ValueError(msg)
        if self.perturbation:
            f.write("  !SetPerturbationBuild\n")
        if self._buffering is False:
            f.write("  !No-buffering\n")
        if self.root_topology is not None:
            f.write(f"  !SetRootTopology={{{self.root_topology}}}\n")
        self.bc.write(f)
