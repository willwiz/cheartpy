import dataclasses as dc
from typing import TextIO
from ...pytools import get_enum, join_fields
from ...interface import *
from ...implementation.problems import BoundaryCondition


@dc.dataclass(slots=True)
class FSExpr:
    var: _Variable | _Expression
    expr: _Expression | _Variable | float = 1
    op: str|None = None

    def to_str(self) -> str:
        return f"{self.var}[{join_fields(self.expr, self.op, char=";")}]"


@dc.dataclass(slots=True)
class FSCouplingTerm:
    test_var: _Variable
    expr: list[FSExpr]


class FSCouplingProblem(_Problem):
    name: str
    space: _Variable
    root_topology: _CheartTopology
    lm: FSCouplingTerm
    terms: dict[str, FSCouplingTerm]
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
        self.terms[str(var)] = FSCouplingTerm(var, list(expr))

    def get_variables(self) -> dict[str, _Variable]:
        vars: dict[str, _Variable] = {str(self.space): self.space}
        vars[str(self.lm.test_var)] = self.lm.test_var
        for t in self.lm.expr:
            if isinstance(t.expr, _Variable):
                vars[str(t.expr)] = t.expr
        for k, v in self.terms.items():
            vars[k] = v.test_var
            for t in v.expr:
                if isinstance(t.var, _Variable):
                    vars[k] = t.var
                if isinstance(t.expr, _Variable):
                    vars[str(t.expr)] = t.expr
        return vars


    def get_aux_vars(self) -> dict[str, _Variable]:
        return self.aux_vars

    def add_aux_vars(self, *var: _Variable) -> None:
        for v in var:
            self.aux_vars[str(v)] = v

    def get_aux_expr(self) -> dict[str, _Expression]:
        return self.aux_expr

    def add_aux_expr(self, *expr: _Expression) -> None:
        for v in expr:
            self.aux_expr[str(v)] = v

    def get_bc_patches(self) -> list[_BCPatch]:
        patches = self.bc.get_patches()
        return [] if patches is None else patches

    def __init__(
        self,
        name: str,
        space: _Variable,
        root_top: _CheartTopology,
    ) -> None:
        self.name = name
        self.space = space
        self.root_topology = root_top
        self.aux_vars = dict()
        self.aux_expr = dict()
        self.terms = dict()
        self.bc = BoundaryCondition()

    def write(self, f: TextIO):
        f.write(f"!DefProblem={{{self.name}|{self.problem}}}\n")
        f.write(f"  !UseVariablePointer={{Space|{self.space}}}\n")
        for t in self.terms.values():
            f.write(f"  !Addterms={{TestVariable[{t.test_var}]|{" ".join([s.to_str() for s in t.expr])}}}\n")
        f.write(f"  !Addterms={{TestVariable[{self.lm.test_var}*]|{" ".join([s.to_str() for s in self.lm.expr])}}}\n")
        if self.perturbation:
            f.write(f"  !SetPerturbationBuild\n")
        f.write(f"  !SetRootTopology={{{self.root_topology}}}\n")
        self.bc.write(f)
