#!/usr/bin/env python3
from typing import Mapping, Sequence

from ..cheart_core.implementation import Expression
from .types import CLTopology, CLBasisExpressions
from ..cheart_core.physics.fs_coupling.fs_coupling_problem import (
    FSCouplingProblem,
    FSExpr,
)
from ..cheart_core.interface import IVariable, ICheartTopology, IExpression


def create_cl_coupling_problem(
    name: str,
    lm: IVariable,
    top: ICheartTopology,
    space: IVariable,
    disp: IVariable,
    motion: IVariable | None,
    p_basis: IExpression,
    m_basis: IExpression,
    neighbours: Sequence[IVariable],
) -> FSCouplingProblem:
    zero_1_expr = Expression(f"zero_1_expr", [0])
    fsbc = FSCouplingProblem(f"PB_{name}", space, top)
    fsbc.perturbation = True
    if motion is None:
        fsbc.set_lagrange_mult(lm, FSExpr(disp, p_basis))
    else:
        fsbc.set_lagrange_mult(lm, FSExpr(disp, p_basis), FSExpr(motion, m_basis))
        fsbc.add_var_deps(motion)
    for v in neighbours:
        fsbc.add_term(v, FSExpr(lm, zero_1_expr)) if str(v) != str(lm) else ...
    # fsbc.set_lagrange_mult(lm, FSExpr(disp, p_basis))
    fsbc.add_term(disp, FSExpr(lm, p_basis))
    fsbc.add_expr_deps(zero_1_expr, p_basis, m_basis)
    return fsbc


def create_cl_coupling_problems(
    tops: Mapping[int, ICheartTopology],
    lms: Mapping[int, IVariable],
    space: IVariable,
    disp: IVariable,
    motion: IVariable | None,
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
            [lms[n] for n in [i - 1, i + 1] if n in lms],
        )
        for i, s in cl_part.node_prefix.items()
    }
    return res
