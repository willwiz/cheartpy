from typing import Sequence
from ..cheart_core.interface import *
from ..cheart_core.implementation import Expression
from ..cheart_core.physics.fs_coupling.fs_coupling_problem import (
    FSCouplingProblem,
    FSExpr,
)
from .types import *


def create_center_pos_expr(
    prefix: str, cl_nodes: Mapping[int, _Variable], p_basis: Mapping[int, _Expression]
):
    cl_pos_expr = Expression(
        prefix,
        [
            " + ".join([f"{cl_nodes[k]}.{j}*{v}" for k, v in p_basis.items()])
            for j in [1, 2, 3]
        ],
    )
    cl_pos_expr.add_var_deps(*cl_nodes.values())
    cl_pos_expr.add_expr_deps(*p_basis.values())
    return cl_pos_expr


def create_cl_position_problem(
    name: str,
    top: _CheartTopology,
    space: _Variable,
    disp: _Variable,
    cl_nodes: _Variable,
    cl_pos_expr: _Expression,
    neighbours: Sequence[_Variable],
    p_basis: _Expression,
) -> FSCouplingProblem:
    zero_expr = Expression(f"zero_1_expr", [0])
    cons = Expression(
        f"{name}_cons_expr", [f"- {cl_pos_expr}.{j} * {p_basis}" for j in [1, 2, 3]]
    )
    cons.add_expr_deps(zero_expr, cl_pos_expr)
    fsbc = FSCouplingProblem(name, space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(
        cl_nodes,
        FSExpr(space, p_basis),
        FSExpr(disp, p_basis),
        FSExpr(cons),
    )
    for v in neighbours:
        fsbc.add_term(v, FSExpr(cl_nodes, zero_expr))
    fsbc.add_expr_deps(p_basis, cons)
    fsbc.add_var_deps(cl_nodes)
    return fsbc


def create_cl_position_problems(
    space: _Variable,
    disp: _Variable,
    cl_nodes: Mapping[int, _Variable],
    tops: Mapping[int, _CheartTopology],
    cl_pos_expr: _Expression,
    p_basis: Mapping[int, _Expression],
) -> Mapping[int, FSCouplingProblem]:
    return {
        k: create_cl_position_problem(
            f"PB_CL{k}Pos",
            v,
            space,
            disp,
            cl_nodes[k],
            cl_pos_expr,
            [cl_nodes[n] for n in [k - 1, k + 1] if n in cl_nodes],
            p_basis[k],
        )
        for k, v in tops.items()
    }
