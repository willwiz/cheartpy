from typing import Mapping
from .types import CLTopology, CLBasisExpressions
from ..cheart_core.physics.fs_coupling.fs_coupling_problem import (
    FSCouplingProblem,
    FSExpr,
)
from ..cheart_core.interface import _Variable, _CheartTopology, _Expression
from ..cheart_core.implementation.expressions import Expression


def create_radial_len_expr(
    space: _Variable,
    disp: _Variable,
    motion: _Variable,
    cl: _Expression,
):
    v_disp = Expression(
        "v_disp_expr", [f"{disp}.{i} + {space}.{i} - {cl}.{i}" for i in [1, 2, 3]]
    )
    r_disp = Expression(
        "r_disp_expr",
        [f"sqrt({" + ".join([f"{v_disp}.{i}*{v_disp}.{i}" for i in [1,2,3]])})"],
    )
    r_disp.add_expr_deps(cl, v_disp)
    r_disp.add_var_deps(disp)
    v_motion = Expression(
        "v_motion_expr", [f"{motion}.{i} + {space}.{i} - {cl}.{i}" for i in [1, 2, 3]]
    )
    r_motion = Expression(
        "r_motion_expr",
        [f"sqrt({" + ".join([f"{v_motion}.{i}*{v_motion}.{i}" for i in [1,2,3]])})"],
    )
    r_motion.add_expr_deps(v_motion)
    r_motion.add_var_deps(motion)
    return r_disp, r_motion


def create_outward_normal_expr(space: _Variable, cl: _Expression):
    n_expr = Expression(
        "Space_p_normal_expr", [f"{space}.{i} - {cl}.{i}" for i in [1, 2, 3]]
    )
    n_expr.add_deps(space, cl)
    m_expr = Expression("Space_m_normal_expr", [f"-{n_expr}.{i}" for i in [1, 2, 3]])
    m_expr.add_deps(n_expr)
    return n_expr, m_expr


def create_dilation_expr(
    disp: _Expression,
    motion: _Expression,
    cl_basis: _Expression,
    k: int,
):
    cons_expr = Expression(f"DL{k}_cons_expr", [f"({disp} - {motion}) * {cl_basis}"])
    cons_expr.add_expr_deps(disp, motion)
    return cons_expr


def create_dilation_expr_terms(
    disp: _Expression, motion: _Expression, cl_basis: CLBasisExpressions
):
    return {
        k: create_dilation_expr(disp, motion, v, k)
        for k, v in cl_basis["pelem"].items()
    }


def create_dilation_problem(
    name: str,
    top: _CheartTopology,
    space: _Variable,
    disp: _Variable,
    motion: _Variable,
    lm: _Variable,
    zeros: _Expression,
    normal: _Expression,
    mormal: _Expression,
) -> FSCouplingProblem:
    fsbc = FSCouplingProblem(name, space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(disp, normal), FSExpr(motion, mormal))
    fsbc.add_term(disp, FSExpr(lm, zeros))
    fsbc.add_expr_deps(zeros, normal, mormal)
    return fsbc


def create_dilation_problems(
    tops: Mapping[int, _CheartTopology],
    space: _Variable,
    disp: _Variable,
    motion: _Variable,
    lms: Mapping[int, _Variable],
    cl_pos_expr: _Expression,
    cl_basis: CLBasisExpressions,
    cl_part: CLTopology,
    dirichlet_bc: bool = True,
) -> dict[int, FSCouplingProblem]:
    zero_expr = Expression(f"zero_expr", [0 for _ in range(3)])
    # cl_pos_expr = create_center_pos_expr(cl_nodes, cl_basis)
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
        for i, s in cl_part.node_prefix.items()
    }
    if dirichlet_bc:
        keys = sorted(res.keys())
        res = {k: res[k] for k in keys[1:-1]}
    # fs1 = next(iter(res.values()))
    # fs1.add_aux_expr(cl_pos_expr)
    return res
