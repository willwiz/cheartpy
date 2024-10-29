from ..cheart_core.interface import *
from ..cheart_core.implementation import Expression
from ..cheart_core.physics.fs_coupling.fs_coupling_problem import (
    FSCouplingProblem,
    FSExpr,
)
from .types import *


def create_center_pos_expr(cl_nodes: _Variable, p_basis: Mapping[int, _Expression]):
    cl_pos_expr = Expression(
        f"cl_pos_expr",
        [
            " + ".join([f"{cl_nodes}.{3*k + j}*{v}" for k, v in p_basis.items()])
            for j in [1, 2, 3]
        ],
    )
    cl_pos_expr.add_var_deps(cl_nodes)
    cl_pos_expr.add_expr_deps(*p_basis.values())
    return cl_pos_expr


def create_cl_position_problem(
    name: str,
    top: _CheartTopology,
    space: _Variable,
    disp: _Variable,
    cl_nodes: _Variable,
    cl_pos_expr: _Expression,
    lm: _Variable,
    p_basis: _Expression,
    i: int,
) -> FSCouplingProblem:
    zeros = Expression(
        f"zeros_{3*cl_nodes.get_dim()}", [0 for _ in range(3 * cl_nodes.get_dim())]
    )
    cons = Expression(
        f"cl{i}_pos_cons", [f"- {cl_pos_expr}.{j} * {p_basis}" for j in [1, 2, 3]]
    )
    fsbc = FSCouplingProblem(name, space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(
        lm,
        FSExpr(space, p_basis),
        FSExpr(disp, p_basis),
        FSExpr(cons),
    )
    fsbc.add_term(cl_nodes, FSExpr(lm, zeros))
    fsbc.add_aux_expr(zeros, p_basis, cl_pos_expr, cons)
    fsbc.add_aux_vars(cl_nodes, lm)
    return fsbc


def create_cl_position_problems(
    space: _Variable,
    disp: _Variable,
    cl_nodes: _Variable,
    cl_pos_expr: _Expression,
    tops: Mapping[int, _CheartTopology],
    lms: Mapping[int, _Variable],
    p_basis: Mapping[int, _Expression],
) -> Mapping[int, FSCouplingProblem]:
    return {
        k: create_cl_position_problem(
            f"PB_CL{k}Pos",
            v,
            space,
            disp,
            cl_nodes,
            cl_pos_expr,
            lms[k],
            p_basis[k],
            k,
        )
        for k, v in tops.items()
    }
