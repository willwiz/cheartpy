from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cheartpy.fe.trait import IExpression, IProblem, IVariable


def add_statevar(p: IProblem | None, *var: IVariable | IExpression | None) -> None:
    if p is None:
        return
    for v in var:
        p.add_state_variable(v)


def add_var_deps(p: IProblem, *var: IVariable | None) -> None:
    for v in var:
        if v is None:
            continue
        if str(v) not in p.aux_vars:
            p.aux_vars[str(v)] = v


def add_expr_deps(p: IProblem, *expr: IExpression | None) -> None:
    for e in expr:
        if e is None:
            continue
        if str(e) not in p.aux_expr:
            p.aux_expr[str(e)] = e


def get_var_deps(p: IProblem) -> dict[str, IVariable]:
    _vars_ = p.get_prob_vars()
    _b_vars_ = {str(v): v for v in p.bc.get_vars_deps()}
    return {**_vars_, **_b_vars_, **p.aux_vars}
