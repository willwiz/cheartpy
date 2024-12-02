__all__ = ["create_cl_refcoupling_problem", "create_cl_curcoupling_problem"]
from ..cheart.physics import FSCouplingProblem, FSExpr
from ..cheart.trait import IVariable, ICheartTopology, IExpression
from ..cheart.impl.expressions import Expression


def create_cl_refcoupling_problem(
    prefix: str,
    space: IVariable,
    basis: IExpression,
    lm: IVariable,
    disp: IVariable,
    ref: IVariable | None = None,
    motion: IVariable | None = None,
    top: ICheartTopology | None = None,
) -> FSCouplingProblem:
    var: list[IVariable | None] = [disp, ref, motion]
    integral_expr = Expression(
        f"{prefix}_expr",
        [
            " - ".join([f"{v}.{i + 1}" for v in var if v is not None])
            for i in range(disp.get_dim())
        ],
    )
    integral_expr.add_deps(disp, ref, motion)
    fsbc = FSCouplingProblem(f"P{prefix}", space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(integral_expr, basis))
    fsbc.add_term(disp, FSExpr(lm, basis))
    fsbc.add_state_variable(ref)
    fsbc.add_deps(basis, integral_expr)
    return fsbc


def create_cl_curcoupling_problem(
    suffix: str,
    node: str,
    top: ICheartTopology,
    space: IVariable,
    motion: IVariable | None,
    U0: IVariable,
    Ut: IVariable,
    lm_ref: IVariable,
    lm_cur: IVariable,
    p_basis: IExpression,
    m_basis: IExpression,
) -> FSCouplingProblem:
    fsbc = FSCouplingProblem(f"P{node}{suffix}", space, top)
    fsbc.perturbation = True
    if motion is None:
        fsbc.set_lagrange_mult(lm_cur, FSExpr(Ut, p_basis))
    else:
        fsbc.set_lagrange_mult(lm_cur, FSExpr(Ut, p_basis), FSExpr(motion, m_basis))
        fsbc.add_var_deps(motion)
    # for v in neighbours:
    #     fsbc.add_term(v, FSExpr(lm, zero_1_expr)) if str(v) != str(lm) else ...
    # fsbc.add_term(Ut, FSExpr(lm_cur, p_basis))
    fsbc.add_term(Ut, FSExpr(lm_cur, p_basis))
    fsbc.add_term(space, FSExpr(space, 0))
    # fsbc.add_term(U0, FSExpr(U0, 0))
    # fsbc.add_term(lm_ref, FSExpr(lm_ref, 0))
    # fsbc.add_term(U0, FSExpr(lm_ref, p_basis))
    # fsbc.add_term(U0, FSExpr(lm_ref, p_basis))
    # fsbc.add_term(lm_ref, FSExpr(U0, p_basis))
    fsbc.add_expr_deps(p_basis, m_basis)
    return fsbc
