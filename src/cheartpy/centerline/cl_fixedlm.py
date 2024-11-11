__all__ = ["create_clfixed_coupling_problems"]
from typing import Mapping, Sequence
from ..cheart_core.implementation import Expression
from ..cheart_core.physics import FSCouplingProblem, FSExpr
from ..cheart_core.interface import IVariable, ICheartTopology, IExpression
from .types import CLTopology, CLBasisExpressions


def create_clfixed_coupling_problem(
    name: str,
    lm: IVariable,
    top: ICheartTopology,
    space: IVariable,
    disp: IVariable,
    lm_fixed: IVariable,
    p_basis: IExpression,
    m_basis: IExpression,
    neighbours: Sequence[IVariable],
) -> FSCouplingProblem:
    zero_1_expr = Expression(f"zero_1_expr", [0])
    fsbc = FSCouplingProblem(f"PB_{name}", space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(lm, p_basis), FSExpr(lm_fixed, m_basis))
    fsbc.add_var_deps(lm_fixed)
    for v in neighbours:
        fsbc.add_term(v, FSExpr(lm, zero_1_expr)) if str(v) != str(lm) else ...
    # fsbc.set_lagrange_mult(lm, FSExpr(disp, p_basis))
    fsbc.add_term(disp, FSExpr(lm, p_basis))
    fsbc.add_expr_deps(zero_1_expr, p_basis, m_basis)
    return fsbc


def create_clfixed_coupling_problems(
    tops: Mapping[int, ICheartTopology],
    lms: Mapping[int, IVariable],
    space: IVariable,
    disp: IVariable,
    lm_fixed: Mapping[int, IVariable],
    cl_part: CLTopology,
    cl_top: CLBasisExpressions,
) -> Mapping[int, FSCouplingProblem]:
    res = {
        i: create_clfixed_coupling_problem(
            s,
            lms[i],
            tops[i],
            space,
            disp,
            lm_fixed[i],
            cl_top["pelem"][i],
            cl_top["melem"][i],
            [lms[n] for n in [i - 1, i + 1] if n in lms],
        )
        for i, s in cl_part.node_prefix.items()
    }
    return res
