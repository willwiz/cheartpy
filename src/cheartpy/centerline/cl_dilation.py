from typing import Mapping
from .types import CLTopology, CLBasisExpressions
from ..cheart_core.physics.fs_coupling.fs_coupling_problem import (
    FSCouplingProblem,
    FSExpr,
)
from ..cheart_core.interface import _Variable, _CheartTopology, _Expression
from ..cheart_core.implementation.expressions import Expression


def create_center_pos_expr(cl_nodes: _Variable, cl_basis: CLBasisExpressions):
    return Expression(
        f"cl_pos_expr",
        [
            "+".join(
                [f"{cl_nodes}.{3*k + j} * {v}" for k, v in cl_basis["pelem"].items()]
            )
            for j in [1, 2, 3]
        ],
    )


def create_dilation_expr(
    disp: _Variable,
    motion: _Variable,
    cl: _Expression,
    p_basis: _Expression,
):
    v_disp = Expression("v_disp_expr", [f"{disp}.{i} - {cl}.{i}" for i in [1, 2, 3]])
    r_disp = Expression(
        "r_disp_expr",
        [f"sqrt({" + ".join([f"{v_disp}.{i}*{v_disp}.{i}" for i in [1,2,3]])})"],
    )
    v_motion = Expression(
        "v_motion_expr", [f"{motion}.{i} - {cl}.{i}" for i in [1, 2, 3]]
    )
    r_motion = Expression(
        "r_motion_expr",
        [f"sqrt({" + ".join([f"{v_motion}.{i}*{v_motion}.{i}" for i in [1,2,3]])})"],
    )
    cons_expr = Expression("dl_cons_expr", [f"({r_disp} - {r_motion}) * {p_basis}"])
    cons_expr.add_var_deps(disp, motion)
    cons_expr.add_expr_deps(r_disp, v_disp, r_motion, v_motion, cl, p_basis)
    return cons_expr


def create_dilation_expr_terms(
    disp: _Variable, motion: _Variable, cl: _Expression, cl_basis: CLBasisExpressions
):
    return {
        k: create_dilation_expr(disp, motion, cl, v)
        for k, v in cl_basis["pelem"].items()
    }


def create_dilation_problem(
    name: str,
    top: _CheartTopology,
    space: _Variable,
    disp: _Variable,
    lm: _Variable,
    zeros: _Expression,
    cons_expr: _Expression,
) -> FSCouplingProblem:
    fsbc = FSCouplingProblem(name, space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(cons_expr))
    fsbc.add_term(disp, FSExpr(lm, zeros))
    fsbc.add_aux_expr(cons_expr)
    return fsbc


def create_dilation_problems(
    tops: Mapping[int, _CheartTopology],
    space: _Variable,
    disp: _Variable,
    motion: _Variable,
    lms: Mapping[int, _Variable],
    cl_nodes: _Variable,
    cl_basis: CLBasisExpressions,
    cl_part: CLTopology,
    dirichlet_bc: bool = True,
) -> dict[int, FSCouplingProblem]:
    zero_expr = Expression(f"zero_expr", [0 for _ in range(3)])
    cl_pos_expr = create_center_pos_expr(cl_nodes, cl_basis)
    cons_exprs = create_dilation_expr_terms(disp, motion, cl_pos_expr, cl_basis)
    res = {
        i: create_dilation_problem(
            f"PB_{s}_DL",
            tops[i],
            space,
            disp,
            lms[i],
            zero_expr,
            cons_exprs[i],
        )
        for i, s in cl_part.node_prefix.items()
    }
    if dirichlet_bc:
        keys = sorted(res.keys())
        res = {k: res[k] for k in keys[1:-1]}
    fs1 = next(iter(res.values()))
    fs1.add_aux_expr(zero_expr, cl_pos_expr)
    return res
