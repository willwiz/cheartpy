__all__ = ["create_cl_refcoupling_problem", "create_cl_curcoupling_problem"]
from ..cheart.physics import FSCouplingProblem, FSExpr
from ..cheart.trait import IVariable, ICheartTopology, IExpression


def create_cl_refcoupling_problem(
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
        fsbc.set_lagrange_mult(lm_ref, FSExpr(U0, p_basis))
    else:
        fsbc.set_lagrange_mult(lm_ref, FSExpr(U0, p_basis))
        # fsbc.set_lagrange_mult(lm_ref, FSExpr(U0, p_basis), FSExpr(motion, m_basis))
        fsbc.add_var_deps(motion)
    # for v in neighbours:
    #     fsbc.add_term(v, FSExpr(lm, zero_1_expr)) if str(v) != str(lm) else ...
    # fsbc.add_term(Ut, FSExpr(lm_cur, p_basis), FSExpr(lm_ref, m_basis))
    fsbc.add_term(U0, FSExpr(lm_ref, p_basis))
    # fsbc.add_term(space, FSExpr(space, 0))
    # fsbc.add_term(U0, FSExpr(lm_ref, p_basis))
    # fsbc.add_term(lm_ref, FSExpr(U0, p_basis))
    fsbc.add_expr_deps(p_basis, m_basis)
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
