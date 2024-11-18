__all__ = ["create_cl_coupling_problem", "create_cl_coupling_problems"]
from typing import Mapping
from ..cheart.physics import FSCouplingProblem, FSExpr
from ..cheart.trait import IVariable, ICheartTopology, IExpression
from .types import CLPartition, CLBasisExpressions


def create_cl_coupling_problem(
    suffix: str,
    node: str,
    top: ICheartTopology,
    space: IVariable,
    disp: IVariable,
    motion: IVariable | None,
    lm: IVariable,
    p_basis: IExpression,
    m_basis: IExpression,
) -> FSCouplingProblem:
    fsbc = FSCouplingProblem(f"P{node}{suffix}", space, top)
    fsbc.perturbation = True
    if motion is None:
        fsbc.set_lagrange_mult(lm, FSExpr(disp, p_basis))
    else:
        fsbc.set_lagrange_mult(lm, FSExpr(disp, p_basis), FSExpr(motion, m_basis))
        fsbc.add_var_deps(motion)
    # for v in neighbours:
    #     fsbc.add_term(v, FSExpr(lm, zero_1_expr)) if str(v) != str(lm) else ...
    # fsbc.set_lagrange_mult(lm, FSExpr(disp, p_basis))
    fsbc.add_term(disp, FSExpr(lm, p_basis))
    fsbc.add_expr_deps(p_basis, m_basis)
    return fsbc


def create_cl_coupling_problems(
    tops: Mapping[int, ICheartTopology],
    lms: Mapping[int, IVariable],
    space: IVariable,
    disp: IVariable,
    motion: IVariable | None,
    cl_part: CLPartition,
    cl_top: CLBasisExpressions,
) -> Mapping[int, FSCouplingProblem]:
    res = {
        i: create_cl_coupling_problem(
            "LM",
            s,
            tops[i],
            space,
            disp,
            motion,
            lms[i],
            cl_top["p"][i],
            cl_top["m"][i],
            # [lms[n] for n in [i - 1, i + 1] if n in lms],
        )
        for i, s in cl_part.n_prefix.items()
    }
    return res
