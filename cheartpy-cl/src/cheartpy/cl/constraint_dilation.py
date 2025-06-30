__all__ = ["create_cl_dilation_constraint_problem"]

from cheartpy.cheart.api import create_expr
from cheartpy.cheart.physics.fs_coupling.struct import FSCouplingProblem, FSExpr
from cheartpy.cheart.trait import IVariable

from .struct import CLTopology


def create_cl_dilation_constraint_problem(
    cl: CLTopology | None,
    space: IVariable,
    lm: IVariable | None,
    disp: IVariable,
    *motion: IVariable | None,
    sfx: str = "DM",
) -> FSCouplingProblem | None:
    if cl is None or lm is None:
        return None
    var: list[IVariable | None] = [disp, *motion]
    integral_expr = create_expr(
        f"{cl}{sfx}_expr",
        [
            f"{cl.elem}.{k + 1} * {cl.basis} * ({' - '.join([f'{v}.{i + 1}' for v in var if v])})"
            for k in range(cl.nn)
            for i in range(disp.get_dim())
        ],
    )
    integral_expr.add_deps(cl.elem, cl.basis, disp, *motion)
    fsbc = FSCouplingProblem(f"P{cl}{sfx}", space, cl.top_i)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(integral_expr, op="trace"))
    fsbc.add_state_variable(disp)
    fsbc.add_expr_deps(integral_expr)
    return fsbc
