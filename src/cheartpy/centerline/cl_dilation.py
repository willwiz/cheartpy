from typing import Mapping
from .types import CLTopology, CLBasisExpressions
from ..cheart_core.physics.fs_coupling.fs_coupling_problem import (
    FSCouplingProblem,
    FSExpr,
)
from ..cheart_core.interface import _Variable, _CheartTopology, _Expression
from ..cheart_core.implementation.expressions import Expression


def create_dilation_problem(
    name: str,
    top: _CheartTopology,
    space: _Variable,
    disp: _Variable,
    motion: _Variable | None,
    lm: _Variable,
    zeros: _Expression,
    p_basis: _Expression,
    m_basis: _Expression,
) -> FSCouplingProblem:
    fsbc = FSCouplingProblem(name, space, top)
    fsbc.perturbation = True
    if motion is None:
        fsbc.set_lagrange_mult(lm, FSExpr(disp, p_basis))
    else:
        fsbc.set_lagrange_mult(lm, FSExpr(disp, p_basis), FSExpr(motion, m_basis))
    fsbc.add_term(disp, FSExpr(lm, zeros))
    fsbc.add_aux_expr(p_basis, m_basis)
    return fsbc


def create_dilation_problems(
    tops: Mapping[int, _CheartTopology],
    space: _Variable,
    disp: _Variable,
    motion: _Variable | None,
    lms: Mapping[int, _Variable],
    cl_basis: CLBasisExpressions,
    cl_part: CLTopology,
    dirichlet_bc: bool = True,
) -> dict[int, FSCouplingProblem]:
    zero_expr = Expression(f"zero_expr", [0 for _ in range(3)])
    res = {
        i: create_dilation_problem(
            f"PB_{s}_DL",
            tops[i],
            space,
            disp,
            motion,
            lms[i],
            zero_expr,
            cl_basis["pelem"][i],
            cl_basis["melem"][i],
        )
        for i, s in cl_part.node_prefix.items()
    }
    if dirichlet_bc:
        keys = sorted(res.keys())
        res = {k: res[k] for k in keys[1:-1]}
    fs1 = next(iter(res.values()))
    fs1.add_aux_expr(zero_expr)
    return res
