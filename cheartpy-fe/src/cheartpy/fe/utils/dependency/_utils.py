from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cheartpy.fe.trait import IExpression, IProblem, IVariable


def add_var_deps(p: IProblem, *var: IVariable | None) -> None:
    for v in var:
        if v is None:
            continue
        if str(v) not in p.var_deps:
            p.var_deps[str(v)] = v


def add_expr_deps(p: IProblem, *expr: IExpression | None) -> None:
    for e in expr:
        if e is None:
            continue
        if str(e) not in p.expr_deps:
            p.expr_deps[str(e)] = e


def get_var_deps(p: IProblem) -> dict[str, IVariable]:
    _vars_ = p.get_prob_vars()
    _b_vars_ = {str(v): v for v in p.bc.get_vars_deps()}
    return {**_vars_, **_b_vars_, **p.var_deps}
