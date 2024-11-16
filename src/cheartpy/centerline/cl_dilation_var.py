__all__ = ["create_dilation_var_problems"]
from typing import Literal, Mapping
from ..cheart.physics import FSCouplingProblem, FSExpr
from ..cheart.trait import IVariable, ICheartTopology, IExpression
from ..cheart.impl.expressions import Expression
from .types import CLPartition, CLBasisExpressions


def create_outward_normal_expr(
    prefix: str, normal: IVariable, basis: IExpression
) -> Mapping[Literal["p", "m"], IExpression]:
    p_expr = Expression(f"{prefix}_p", [f"{normal}.{i} * {basis}" for i in [1, 2, 3]])
    p_expr.add_deps(normal, basis)
    m_expr = Expression(f"{prefix}_m", [f"-{normal}.{i} * {basis}" for i in [1, 2, 3]])
    m_expr.add_deps(normal, basis)
    return {"p": p_expr, "m": m_expr}


def create_dilation_var_problem(
    name: str,
    top: ICheartTopology,
    space: IVariable,
    disp: IVariable,
    motion: IVariable,
    lm: IVariable,
    zeros: IExpression,
    normal_p: IExpression,
    normal_m: IExpression,
) -> FSCouplingProblem:
    # zero_1_expr = Expression(f"zero_1_expr", [0])
    fsbc = FSCouplingProblem(name, space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(disp, normal_p), FSExpr(motion, normal_m))
    fsbc.add_term(disp, FSExpr(lm, zeros))
    # for v in neighbours:
    #     fsbc.add_term(v, FSExpr(lm, zero_1_expr)) if str(v) != str(lm) else ...
    fsbc.add_expr_deps(zeros, normal_p, normal_m)
    return fsbc


def create_dilation_var_problems(
    space: IVariable,
    disp: IVariable,
    motion: IVariable,
    cl_part: CLPartition,
    cl_basis: CLBasisExpressions,
    tops: Mapping[int, ICheartTopology],
    lms: Mapping[int, IVariable],
    cl_normal: Mapping[int, IVariable],
) -> dict[int, FSCouplingProblem]:
    zero_expr = Expression(f"zero_3_expr", [0 for _ in range(3)])
    normal_expr = {
        k: create_outward_normal_expr(
            f"{v}_normal_expr", cl_normal[k], cl_basis["p"][k]
        )
        for k, v in cl_part.n_prefix.items()
    }
    res = {
        k: create_dilation_var_problem(
            f"PB{v}_DL",
            tops[k],
            space,
            disp,
            motion,
            lms[k],
            zero_expr,
            normal_expr[k]["p"],
            normal_expr[k]["m"],
            # [lms[n] for n in [k - 1, k + 1] if n in lms],
        )
        for k, v in cl_part.n_prefix.items()
    }
    return res
