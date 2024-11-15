__all__ = ["FSCouplingProblem", "FSExpr"]
import dataclasses as dc
from typing import Sequence, TextIO, ValuesView
from ...pytools import join_fields
from ...trait import *
from ...impl import BoundaryCondition


@dc.dataclass(slots=True)
class FSExpr:
    var: IVariable | IExpression
    mult: IExpression | IVariable | float = 1
    op: str | None = None

    def to_str(self) -> str:
        return f"{self.var}[{join_fields(self.mult, self.op, char=";")}]"


@dc.dataclass(slots=True)
class FSCouplingTerm:
    test_var: IVariable
    terms: list[FSExpr]


class FSCouplingProblem(IProblem):
    name: str
    space: IVariable
    root_topology: ICheartTopology | None
    lm: FSCouplingTerm
    m_terms: dict[str, FSCouplingTerm]
    bc: IBoundaryCondition
    aux_vars: dict[str, IVariable]
    aux_expr: dict[str, IExpression]
    problem: str = "fscoupling_problem"
    perturbation: bool = True

    def __repr__(self) -> str:
        return self.name

    def set_lagrange_mult(self, var: IVariable, *expr: FSExpr) -> None:
        self.lm = FSCouplingTerm(var, list(expr))

    def add_term(self, var: IVariable, *expr: FSExpr) -> None:
        self.m_terms[str(var)] = FSCouplingTerm(var, list(expr))

    def get_prob_vars(self) -> dict[str, IVariable]:
        vars: dict[str, IVariable] = {str(self.space): self.space}
        vars[str(self.lm.test_var)] = self.lm.test_var
        # for t in self.lm.terms:
        #     if isinstance(t.var, _Variable):
        #         if str(t.var) not in vars:
        #             vars[str(t.var)] = t.var
        for _, v in self.m_terms.items():
            if str(v.test_var) not in vars:
                vars[str(v.test_var)] = v.test_var
            # for t in v.terms:
            #     if isinstance(t.var, _Variable):
            #         vars[str(t.var)] = t.var
            # if isinstance(t.expr, _Variable):
            #     vars[str(t.expr)] = t.expr
        # for v in self.bc.get_vars_deps():
        #     vars[str(v)] = v
        return vars

    def add_var_deps(self, *var: IVariable) -> None:
        for v in var:
            if str(v) not in self.aux_vars:
                self.aux_vars[str(v)] = v

    def add_expr_deps(self, *expr: IExpression) -> None:
        for v in expr:
            if str(v) not in self.aux_expr:
                self.aux_expr[str(v)] = v

    def get_var_deps(self) -> ValuesView[IVariable]:
        _vars_ = {str(v): v for v in self.bc.get_vars_deps()}
        if str(self.space) not in _vars_:
            _vars_[str(self.space)] = self.space
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

    def __init__(
        self,
        name: str,
        space: IVariable,
        root_top: ICheartTopology | None = None,
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
