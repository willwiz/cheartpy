__all__ = ["create_dilation_ref_problem"]
from .data import CLTopology
from ..cheart.physics import FSCouplingProblem, FSExpr
from ..cheart.trait import IVariable
from ..cheart.impl.expressions import Expression


def create_dilation_ref_problem(
    prefix: str,
    cl: CLTopology,
    space: IVariable,
    lm: IVariable,
    disp: IVariable,
    ref: IVariable | None,
    motion: IVariable | None,
) -> FSCouplingProblem:
    var: list[IVariable | None] = [disp, ref, motion]
    integral_expr = Expression(
        f"{prefix}_expr",
        [
            f"{cl.elem}.{k + 1} * {cl.basis} * ({" - ".join([f"{v}.{i + 1}" for v in var if v is not None])})"
            for k in range(cl.nn)
            for i in range(disp.get_dim())
        ],
    )
    integral_expr.add_deps(cl.elem, cl.basis, disp, ref, motion)
    fsbc = FSCouplingProblem(f"P{prefix}", space, cl.top_i)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(integral_expr, op="trace"))
    fsbc.add_state_variable(disp, ref, space)
    fsbc.add_expr_deps(integral_expr)
    return fsbc
