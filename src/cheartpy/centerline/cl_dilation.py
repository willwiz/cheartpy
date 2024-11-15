__all__ = ["create_dilation_problems"]
from typing import Mapping
from ..cheart.physics import FSCouplingProblem, FSExpr
from ..cheart.trait import IVariable, ICheartTopology, IExpression
from ..cheart.impl.expressions import Expression
from .types import CLPartition, CLBasisExpressions


def create_outward_normal_expr(space: IVariable, cl: IExpression):
    dist_expr = Expression(
        "Space_dl_disp_expr", [f"{space}.{i} - {cl}.{i}" for i in [1, 2, 3]]
    )
    dist_expr.add_deps(space, cl)
    mag = Expression(
        "Space_norm_expr",
        [f"sqrt({'+'.join([f"{dist_expr}.{i}*{dist_expr}.{i}" for i in [1, 2, 3]])})"],
    )
    mag.add_deps(dist_expr)
    p_expr = Expression(
        "Space_p_normal_expr", [f"-{dist_expr}.{i} / {mag}" for i in [1, 2, 3]]
    )
    p_expr.add_deps(dist_expr, mag)
    m_expr = Expression("Space_m_normal_expr", [f"-{p_expr}.{i}" for i in [1, 2, 3]])
    m_expr.add_deps(p_expr)
    return p_expr, m_expr


def create_dilation_problem(
    name: str,
    top: ICheartTopology,
    space: IVariable,
    disp: IVariable,
    motion: IVariable,
    lm: IVariable,
    zeros: IExpression,
    normal: IExpression,
    mormal: IExpression,
) -> FSCouplingProblem:
    fsbc = FSCouplingProblem(name, space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(disp, normal), FSExpr(motion, mormal))
    fsbc.add_term(disp, FSExpr(lm, zeros))
    fsbc.add_expr_deps(zeros, normal, mormal)
    return fsbc


def create_dilation_problems(
    tops: Mapping[int, ICheartTopology],
    space: IVariable,
    disp: IVariable,
    motion: IVariable,
    lms: Mapping[int, IVariable],
    cl_normal: Mapping[int, IVariable],
    cl_part: CLPartition,
    cl_basis: CLBasisExpressions,
    cl_pos_expr: IExpression,
) -> dict[int, FSCouplingProblem]:
    zero_expr = Expression(f"zero_expr", [0 for _ in range(3)])
    p_norm_expr, m_norm_expr = create_outward_normal_expr(space, cl_pos_expr)
    res = {
        i: create_dilation_problem(
            f"PB_{s}_DL",
            tops[i],
            space,
            disp,
            motion,
            lms[i],
            zero_expr,
            p_norm_expr,
            m_norm_expr,
        )
        for i, s in cl_part.n_prefix.items()
    }
    return res
