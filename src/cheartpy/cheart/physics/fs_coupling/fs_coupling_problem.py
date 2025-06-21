__all__ = ["FSCouplingProblem", "FSExpr"]
import dataclasses as dc
from collections.abc import Sequence, ValuesView
from typing import Literal, TextIO

from ...api import create_bc
from ...pytools import join_fields
from ...trait import *


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


@dc.dataclass(slots=True)
class FSCouplingTerm:
    test_var: IVariable
    terms: list[FSExpr]


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
        self.m_terms = dict()
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
        vars: dict[str, IVariable] = {str(self.space): self.space}
        if self.lm is not None:
            vars[str(self.lm.test_var)] = self.lm.test_var
        for _, v in self.m_terms.items():
            if str(v.test_var) not in vars:
                vars[str(v.test_var)] = v.test_var
        return vars

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
        _vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        if str(self.space) not in _vars_:
            _vars_[str(self.space)] = self.space
        if self.lm is not None:
            _vars_[str(self.lm.test_var)] = self.lm.test_var
            for t in self.lm.terms:
                if isinstance(t.var, IVariable):
                    if str(t.var) not in _vars_:
                        _vars_[str(t.var)] = t.var
                if isinstance(t.mult, IVariable):
                    if str(t.mult) not in _vars_:
                        _vars_[str(t.mult)] = t.mult
        for v in self.m_terms.values():
            if str(v.test_var) not in _vars_:
                _vars_[str(v.test_var)] = v.test_var
            for t in v.terms:
                if isinstance(t.var, IVariable):
                    if str(t.var) not in _vars_:
                        _vars_[str(t.var)] = t.var
                if isinstance(t.mult, IVariable):
                    if str(t.mult) not in _vars_:
                        _vars_[str(t.mult)] = t.mult
        return {**self.aux_vars, **_vars_}.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_ = {str(e): e for e in self.bc.get_expr_deps()}
        if self.lm is not None:
            for t in self.lm.terms:
                if isinstance(t.var, IExpression):
                    if str(t.var) not in _expr_:
                        _expr_[str(t.var)] = t.var
                if isinstance(t.mult, IExpression):
                    if str(t.mult) not in _expr_:
                        _expr_[str(t.mult)] = t.mult
        for v in self.m_terms.values():
            for t in v.terms:
                if isinstance(t.var, IExpression):
                    if str(t.var) not in _expr_:
                        _expr_[str(t.var)] = t.var
                if isinstance(t.mult, IExpression):
                    if str(t.mult) not in _expr_:
                        _expr_[str(t.mult)] = t.mult
        return {**self.aux_expr, **_expr_}.values()

    def get_bc_patches(self) -> Sequence[IBCPatch]:
        patches = self.bc.get_patches()
        return list() if patches is None else list(patches)

    def write(self, f: TextIO):
        f.write(f"!DefProblem={{{self}|{self._problem_name}}}\n")
        f.write(f"  !UseVariablePointer={{Space|{self.space}}}\n")
        f.writelines(
            f"  !Addterms={{TestVariable[{t.test_var}]|{' '.join([s.to_str() for s in t.terms])}}}\n"
            for t in self.m_terms.values()
        )
        if self.lm is not None:
            f.write(
                f"  !Addterms={{TestVariable[{self.lm.test_var}*]|{' '.join([s.to_str() for s in self.lm.terms])}}}\n",
            )
        else:
            raise ValueError(f"Lagrange multiplier not set for {self}")
        if self.perturbation:
            f.write("  !SetPerturbationBuild\n")
        if self._buffering is False:
            f.write("  !No-buffering\n")
        if self.root_topology is not None:
            f.write(f"  !SetRootTopology={{{self.root_topology}}}\n")
        self.bc.write(f)
