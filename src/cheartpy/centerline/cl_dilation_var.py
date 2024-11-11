from typing import Literal, Mapping, Sequence
from .types import CLTopology, CLBasisExpressions
from ..cheart_core.physics.fs_coupling.fs_coupling_problem import (
    FSCouplingProblem,
    FSExpr,
)
from ..cheart_core.interface import IVariable, ICheartTopology, IExpression
from ..cheart_core.implementation.expressions import Expression


def create_outward_normal_expr(
    prefix: str, normal: IVariable, basis: IExpression
) -> Mapping[Literal["p", "m"], IExpression]:
    p_expr = Expression(f"{prefix}_p", [f"{normal}.{i} * {basis}" for i in [1, 2, 3]])
    p_expr.add_deps(normal)
    m_expr = Expression(f"{prefix}_m", [f"-{normal}.{i} * {basis}" for i in [1, 2, 3]])
    m_expr.add_deps(normal)
    return {"p": p_expr, "m": m_expr}


def create_dilation_var_problem(
    name: str,
    top: ICheartTopology,
    space: IVariable,
    disp: IVariable,
    motion: IVariable,
    lm: IVariable,
    zeros: IExpression,
    normal: Mapping[Literal["p", "m"], IExpression],
    neighbours: Sequence[IVariable],
) -> FSCouplingProblem:
    zero_1_expr = Expression(f"zero_1_expr", [0])
    fsbc = FSCouplingProblem(name, space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(disp, normal["p"]), FSExpr(motion, normal["m"]))
    fsbc.add_term(disp, FSExpr(lm, zeros))
    for v in neighbours:
        fsbc.add_term(v, FSExpr(lm, zero_1_expr)) if str(v) != str(lm) else ...
    fsbc.add_expr_deps(zeros, zero_1_expr, normal["p"], normal["m"])
    return fsbc


def create_dilation_var_problems(
    tops: Mapping[int, ICheartTopology],
    space: IVariable,
    disp: IVariable,
    motion: IVariable,
    lms: Mapping[int, IVariable],
    cl_normal: Mapping[int, IVariable],
    cl_part: CLTopology,
    cl_basis: CLBasisExpressions,
    cl_pos_expr: IExpression,
) -> dict[int, FSCouplingProblem]:
    zero_expr = Expression(f"zero_3_expr", [0 for _ in range(3)])
    normal_expr = {
        k: create_outward_normal_expr(
            f"{v}_normal_expr", cl_normal[k], cl_basis["pelem"][k]
        )
        for k, v in cl_part.node_prefix.items()
    }
    res = {
        k: create_dilation_var_problem(
            f"PB_{v}_DL",
            tops[k],
            space,
            disp,
            motion,
            lms[k],
            zero_expr,
            normal_expr[k],
            [lms[n] for n in [k - 1, k + 1] if n in lms],
        )
        for k, v in cl_part.node_prefix.items()
    }
    return res
