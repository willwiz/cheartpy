#!/usr/bin/env python3
from typing import Mapping
from .types import CLTopology, CLBasisExpressions
from ..cheart_core.physics.fs_coupling.fs_coupling_problem import (
    FSCouplingProblem,
    FSExpr,
)
from ..cheart_core.interface import _Variable, _CheartTopology, _Expression


def create_cl_coupling_problem(
    name: str,
    lm: _Variable,
    top: _CheartTopology,
    space: _Variable,
    disp: _Variable,
    motion: _Variable | None,
    p_basis: _Expression,
    m_basis: _Expression,
) -> FSCouplingProblem:
    fsbc = FSCouplingProblem(f"PB_{name}", space, top)
    fsbc.perturbation = True
    if motion is None:
        fsbc.set_lagrange_mult(lm, FSExpr(disp, p_basis))
    else:
        fsbc.set_lagrange_mult(lm, FSExpr(disp, p_basis), FSExpr(motion, m_basis))
    # fsbc.set_lagrange_mult(lm, FSExpr(disp, p_basis))
    fsbc.add_term(disp, FSExpr(lm, p_basis))
    fsbc.add_aux_expr(p_basis, m_basis)
    return fsbc


def create_cl_coupling_problems(
    tops: Mapping[int, _CheartTopology],
    lms: Mapping[int, _Variable],
    space: _Variable,
    disp: _Variable,
    motion: _Variable | None,
    cl_part: CLTopology,
    cl_top: CLBasisExpressions,
) -> Mapping[int, FSCouplingProblem]:
    res = {
        i: create_cl_coupling_problem(
            s,
            lms[i],
            tops[i],
            space,
            disp,
            motion,
            cl_top["pelem"][i],
            cl_top["melem"][i],
        )
        for i, s in cl_part.node_prefix.items()
    }
    # next(iter(res.values())).add_aux_vars(motion)
    return res
