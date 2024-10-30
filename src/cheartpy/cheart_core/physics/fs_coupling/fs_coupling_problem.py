import dataclasses as dc
from typing import Sequence, TextIO, ValuesView
from ...pytools import join_fields
from ...interface import *
from ...implementation import BoundaryCondition


@dc.dataclass(slots=True)
class FSExpr:
    var: _Variable | _Expression
    mult: _Expression | _Variable | float = 1
    op: str | None = None

    def to_str(self) -> str:
        return f"{self.var}[{join_fields(self.mult, self.op, char=";")}]"


@dc.dataclass(slots=True)
class FSCouplingTerm:
    test_var: _Variable
    terms: list[FSExpr]


class FSCouplingProblem(_Problem):
    name: str
    space: _Variable
    root_topology: _CheartTopology | None
    lm: FSCouplingTerm
    m_terms: dict[str, FSCouplingTerm]
    bc: _BoundaryCondition
    aux_vars: dict[str, _Variable]
    aux_expr: dict[str, _Expression]
    problem: str = "fscoupling_problem"
    perturbation: bool = True

    def __repr__(self) -> str:
        return self.name

    def set_lagrange_mult(self, var: _Variable, *expr: FSExpr) -> None:
        self.lm = FSCouplingTerm(var, list(expr))

    def add_term(self, var: _Variable, *expr: FSExpr) -> None:
        self.m_terms[str(var)] = FSCouplingTerm(var, list(expr))

    def get_prob_vars(self) -> dict[str, _Variable]:
        vars: dict[str, _Variable] = {str(self.space): self.space}
        vars[str(self.lm.test_var)] = self.lm.test_var
        # for t in self.lm.terms:
        #     if isinstance(t.var, _Variable):
        #         if str(t.var) not in vars:
        #             vars[str(t.var)] = t.var
        for k, v in self.m_terms.items():
            if str(v.test_var) not in vars:
                vars[str(v.test_var)] = v.test_var
            # for t in v.terms:
            #     if isinstance(t.var, _Variable):
            #         vars[str(t.var)] = t.var
            # if isinstance(t.expr, _Variable):
            #     vars[str(t.expr)] = t.expr
        for v in self.bc.get_vars_deps():
            vars[str(v)] = v
        return vars

    def add_var_deps(self, *var: _Variable) -> None:
        for v in var:
            if str(v) not in self.aux_vars:
                self.aux_vars[str(v)] = v

    def add_expr_deps(self, *expr: _Expression) -> None:
        for v in expr:
            if str(v) not in self.aux_expr:
                self.aux_expr[str(v)] = v

    def get_var_deps(self) -> ValuesView[_Variable]:
        vars = {str(v): v for v in self.bc.get_vars_deps()}
        if str(self.space) not in vars:
            vars[str(self.space)] = self.space
        vars[str(self.lm.test_var)] = self.lm.test_var
        for t in self.lm.terms:
            if isinstance(t.var, _Variable):
                if str(t.var) not in vars:
                    vars[str(t.var)] = t.var
            if isinstance(t.mult, _Variable):
                if str(t.mult) not in vars:
                    vars[str(t.mult)] = t.mult
        for k, v in self.m_terms.items():
            if str(v.test_var) not in vars:
                vars[str(v.test_var)] = v.test_var
            for t in v.terms:
                if isinstance(t.var, _Variable):
                    if str(t.var) not in vars:
                        vars[str(t.var)] = t.var
                if isinstance(t.mult, _Variable):
                    if str(t.mult) not in vars:
                        vars[str(t.mult)] = t.mult

        return {**self.aux_vars, **vars}.values()

    def get_expr_deps(self) -> ValuesView[_Expression]:
        _expr_ = {str(e): e for e in self.bc.get_expr_deps()}
        for t in self.lm.terms:
            if isinstance(t.var, _Expression):
                if str(t.var) not in _expr_:
                    _expr_[str(t.var)] = t.var
            if isinstance(t.mult, _Expression):
                if str(t.mult) not in _expr_:
                    _expr_[str(t.mult)] = t.mult
        for k, v in self.m_terms.items():
            for t in v.terms:
                if isinstance(t.var, _Expression):
                    if str(t.var) not in _expr_:
                        _expr_[str(t.var)] = t.var
                if isinstance(t.mult, _Expression):
                    if str(t.mult) not in _expr_:
                        _expr_[str(t.mult)] = t.mult
        return {**self.aux_expr, **_expr_}.values()

    def get_bc_patches(self) -> Sequence[_BCPatch]:
        patches = self.bc.get_patches()
        return list() if patches is None else patches

    def __init__(
        self,
        name: str,
        space: _Variable,
        root_top: _CheartTopology | None = None,
    ) -> None:
        self.name = name
        self.space = space
        self.root_topology = None if root_top is None else root_top
        self.aux_vars = dict()
        self.aux_expr = dict()
        self.m_terms = dict()
        self.bc = BoundaryCondition()

    def write(self, f: TextIO):
        f.write(f"!DefProblem={{{self.name}|{self.problem}}}\n")
        f.write(f"  !UseVariablePointer={{Space|{self.space}}}\n")
        for t in self.m_terms.values():
            f.write(
                f"  !Addterms={{TestVariable[{t.test_var}]|{" ".join([s.to_str() for s in t.terms])}}}\n"
            )
        f.write(
            f"  !Addterms={{TestVariable[{self.lm.test_var}*]|{" ".join([s.to_str() for s in self.lm.terms])}}}\n"
        )
        if self.perturbation:
            f.write(f"  !SetPerturbationBuild\n")
        if self.root_topology is not None:
            f.write(f"  !SetRootTopology={{{self.root_topology}}}\n")
        self.bc.write(f)
