__all__ = ["create_cl_dilation_constraint_problem"]
from .data import CLTopology
from ..cheart.physics import FSCouplingProblem, FSExpr
from ..cheart.trait import IVariable
from ..cheart.api import create_expr


def create_cl_dilation_constraint_problem(
    prefix: str,
    cl: CLTopology | None,
    space: IVariable,
    lm: IVariable | None,
    disp: IVariable,
    *motion: IVariable | None,
) -> FSCouplingProblem | None:
    if cl is None or lm is None:
        return None
    var: list[IVariable | None] = [disp, *motion]
    integral_expr = create_expr(
        f"{prefix}_expr",
        [
            f"{cl.elem}.{k + 1} * {cl.basis} * ({" - ".join([f"{v}.{i + 1}" for v in var if v])})"
            for k in range(cl.nn)
            for i in range(disp.get_dim())
        ],
    )
    integral_expr.add_deps(cl.elem, cl.basis, disp, *motion)
    fsbc = FSCouplingProblem(f"P{prefix}", space, cl.top_i)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(integral_expr, op="trace"))
    fsbc.add_state_variable(space, disp)
    fsbc.add_expr_deps(integral_expr)
    return fsbc
