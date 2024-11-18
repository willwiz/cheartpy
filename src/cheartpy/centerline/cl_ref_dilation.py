__all__ = ["create_dilation_ref_problem"]
from typing import Literal, Mapping
from ..cheart.physics import FSCouplingProblem, FSExpr
from ..cheart.trait import IVariable, ICheartTopology, IExpression
from ..cheart.impl.expressions import Expression


def create_outward_normal_expr(
    prefix: str, normal: IVariable, basis: IExpression
) -> Mapping[Literal["p", "m"], IExpression]:
    p_expr = Expression(f"{prefix}_p", [f"{normal}.{i} * {basis}" for i in [1, 2, 3]])
    p_expr.add_deps(normal, basis)
    m_expr = Expression(f"{prefix}_m", [f"-{normal}.{i} * {basis}" for i in [1, 2, 3]])
    m_expr.add_deps(normal, basis)
    return {"p": p_expr, "m": m_expr}


def create_dilation_ref_problem(
    sfx: str,
    node: str,
    top: ICheartTopology,
    space: IVariable,
    ref_disp: IVariable,
    cur_disp: IVariable,
    motion: IVariable,
    lm: IVariable,
    normal_p: IExpression,
    normal_m: IExpression,
) -> FSCouplingProblem:
    # zero_1_expr = Expression(f"zero_1_expr", [0])
    zero_3_expr = Expression(f"zero_3_expr", [0 for _ in range(3)])
    fsbc = FSCouplingProblem(f"P{node}{sfx}", space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(
        lm,
        FSExpr(cur_disp, normal_p),
        # FSExpr(ref_disp, normal_m),
        FSExpr(motion, normal_m),
    )
    fsbc.add_term(cur_disp, FSExpr(cur_disp, 0))
    fsbc.add_term(ref_disp, FSExpr(ref_disp, 0))
    # fsbc.add_term(space, FSExpr(lm, zero_3_expr))
    # for v in neighbours:
    #     fsbc.add_term(v, FSExpr(lm, zero_1_expr)) if str(v) != str(lm) else ...
    fsbc.add_expr_deps(zero_3_expr, normal_p, normal_m)
    return fsbc
