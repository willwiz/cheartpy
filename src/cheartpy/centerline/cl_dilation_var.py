from typing import Literal, Mapping
from .types import CLTopology, CLBasisExpressions
from ..cheart_core.physics.fs_coupling.fs_coupling_problem import (
    FSCouplingProblem,
    FSExpr,
)
from ..cheart_core.interface import _Variable, _CheartTopology, _Expression
from ..cheart_core.implementation.expressions import Expression


def create_outward_normal_expr(
    prefix: str, normal: _Variable, basis: _Expression
) -> Mapping[Literal["p", "m"], _Expression]:
    p_expr = Expression(f"{prefix}_p", [f"{normal}.{i} * {basis}" for i in [1, 2, 3]])
    p_expr.add_deps(normal)
    m_expr = Expression(f"{prefix}_m", [f"-{normal}.{i} * {basis}" for i in [1, 2, 3]])
    m_expr.add_deps(normal)
    return {"p": p_expr, "m": m_expr}


def create_dilation_problem(
    name: str,
    top: _CheartTopology,
    space: _Variable,
    disp: _Variable,
    motion: _Variable,
    lm: _Variable,
    zeros: _Expression,
    normal: Mapping[Literal["p", "m"], _Expression],
) -> FSCouplingProblem:
    fsbc = FSCouplingProblem(name, space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(disp, normal["p"]), FSExpr(motion, normal["m"]))
    fsbc.add_term(disp, FSExpr(lm, zeros))
    fsbc.add_expr_deps(zeros, normal["p"], normal["m"])
    return fsbc


def create_dilation_problems(
    tops: Mapping[int, _CheartTopology],
    space: _Variable,
    disp: _Variable,
    motion: _Variable,
    lms: Mapping[int, _Variable],
    cl_normal: Mapping[int, _Variable],
    cl_part: CLTopology,
    cl_basis: CLBasisExpressions,
    cl_pos_expr: _Expression,
) -> dict[int, FSCouplingProblem]:
    zero_expr = Expression(f"zero_expr", [0 for _ in range(3)])
    normal_expr = {
        k: create_outward_normal_expr(
            f"{v}_normal_expr", cl_normal[k], cl_basis["pelem"][k]
        )
        for k, v in cl_part.node_prefix.items()
    }
    res = {
        k: create_dilation_problem(
            f"PB_{v}_DL",
            tops[k],
            space,
            disp,
            motion,
            lms[k],
            zero_expr,
            normal_expr[k],
        )
        for k, v in cl_part.node_prefix.items()
    }
    return res
