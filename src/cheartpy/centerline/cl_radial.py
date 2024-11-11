__all__ = ["create_radial_problems"]
from typing import Mapping, Sequence
from ..cheart_core.physics import FSCouplingProblem, FSExpr
from ..cheart_core.interface import IVariable, ICheartTopology, IExpression
from ..cheart_core.implementation.expressions import Expression
from .types import CLTopology, CLBasisExpressions


def create_radial_len_expr(
    space: IVariable,
    disp: IVariable,
    motion: IVariable,
    cl: IExpression,
):
    v_disp = Expression("v_disp_expr", [f"{disp}.{i} - {cl}.{i}" for i in [1, 2, 3]])
    v_disp.add_deps(disp, cl)
    r_disp = Expression(
        "r_disp_expr",
        [f"sqrt({" + ".join([f"{v_disp}.{i}*{v_disp}.{i}" for i in [1,2,3]])})"],
    )
    r_disp.add_deps(v_disp)
    v_motion = Expression(
        "v_motion_expr", [f"{motion}.{i} - {cl}.{i}" for i in [1, 2, 3]]
    )
    v_motion.add_deps(motion, cl)
    r_motion = Expression(
        "r_motion_expr",
        [f"sqrt({" + ".join([f"{v_motion}.{i}*{v_motion}.{i}" for i in [1,2,3]])})"],
    )
    r_motion.add_deps(v_motion)
    return r_disp, r_motion


def create_radial_expr(
    disp: IExpression,
    motion: IExpression,
    cl_basis: IExpression,
    k: int,
):
    cons_expr = Expression(f"DL{k}_cons_expr", [f"({disp} - {motion}) * {cl_basis}"])
    cons_expr.add_expr_deps(disp, motion)
    return cons_expr


def create_radial_expr_terms(
    disp: IExpression, motion: IExpression, cl_basis: CLBasisExpressions
):
    return {
        k: create_radial_expr(disp, motion, v, k) for k, v in cl_basis["pelem"].items()
    }


def create_radial_problem(
    name: str,
    top: ICheartTopology,
    space: IVariable,
    disp: IVariable,
    lm: IVariable,
    zeros: IExpression,
    cons_expr: IExpression,
    neighbours: Sequence[IVariable],
) -> FSCouplingProblem:
    zero_1_expr = Expression(f"zero_1_expr", [0])
    fsbc = FSCouplingProblem(name, space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(cons_expr))
    fsbc.add_term(disp, FSExpr(lm, zeros))
    for v in neighbours:
        fsbc.add_term(v, FSExpr(lm, zero_1_expr)) if str(v) != str(lm) else ...
    fsbc.add_expr_deps(zeros, cons_expr, zero_1_expr)
    return fsbc


def create_radial_problems(
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
    zero_expr = Expression(f"zero_expr", [0 for _ in range(3)])
    r_disp_expr, r_motion_expr = create_radial_len_expr(
        space, disp, motion, cl_pos_expr
    )
    cons_exprs = create_radial_expr_terms(r_disp_expr, r_motion_expr, cl_basis)
    res = {
        k: create_radial_problem(
            f"PB_{v}_DL",
            tops[k],
            space,
            disp,
            lms[k],
            zero_expr,
            cons_exprs[k],
            [lms[n] for n in [k - 1, k + 1] if n in lms],
        )
        for k, v in cl_part.node_prefix.items()
    }
    # if dirichlet_bc:
    #     keys = sorted(res.keys())
    #     res = {k: res[k] for k in keys[1:-1]}
    return res
