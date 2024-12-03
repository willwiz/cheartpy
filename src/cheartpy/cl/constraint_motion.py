__all__ = ["create_cl_motion_constraint_problem"]
from .data import CLTopology
from ..cheart.impl.expressions import Expression
from ..cheart.physics import FSCouplingProblem, FSExpr
from ..cheart.trait import IVariable


def create_cl_motion_constraint_problem(
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
    integral_expr = Expression(
        f"{prefix}_expr",
        [" - ".join([f"{v}.{i + 1}" for v in var if v]) for i in range(disp.get_dim())],
    )
    integral_expr.add_deps(disp, *motion)
    fsbc = FSCouplingProblem(f"P{prefix}", space, cl.top_i)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(integral_expr, cl.basis))
    fsbc.add_term(disp, FSExpr(lm, cl.basis))
    fsbc.add_deps(cl.basis, integral_expr)
    return fsbc
