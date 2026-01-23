from typing import TYPE_CHECKING

from cheartpy.fe.api import create_expr
from cheartpy.fe.physics.fs_coupling import FSCouplingProblem, FSExpr

if TYPE_CHECKING:
    from cheartpy.fe.trait import IVariable

    from .struct import CLStructure


def create_cl_motion_constraint_problem(
    cl: CLStructure | None,
    space: IVariable,
    lm: IVariable | None,
    disp: IVariable,
    *motion: IVariable | None,
    sfx: str = "CL",
) -> FSCouplingProblem | None:
    if cl is None or lm is None:
        return None
    var: list[IVariable | None] = [disp, *motion]
    integral_expr = create_expr(
        f"{cl}{sfx}_expr",
        [" - ".join([f"{v}.{i + 1}" for v in var if v]) for i in range(disp.get_dim())],
    )
    integral_expr.add_deps(disp, *motion)
    fsbc = FSCouplingProblem(f"P{cl}{sfx}", space, cl.top_i)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(integral_expr, cl.basis))
    fsbc.add_term(disp, FSExpr(lm, cl.basis))
    fsbc.add_deps(cl.basis, integral_expr)
    return fsbc
