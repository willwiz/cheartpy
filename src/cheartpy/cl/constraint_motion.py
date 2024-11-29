__all__ = ["create_cl_coupling_problem", "create_cl_coupling_problems"]
from typing import Mapping

from ..cheart.impl.expressions import Expression
from ..cheart.physics import FSCouplingProblem, FSExpr
from ..cheart.trait import IVariable, ICheartTopology, IExpression
from .types import CLPartition, CLBasisExpressions


def create_cl_coupling_problem(
    prefix: str,
    space: IVariable,
    basis: IExpression,
    lm: IVariable,
    disp: IVariable,
    motion: IVariable | IExpression | None = None,
    top: ICheartTopology | None = None,
) -> FSCouplingProblem:
    if motion is None:
        integral_expr = Expression(f"{prefix}_expr", [f"{disp}"])
    else:
        integral_expr = Expression(f"{prefix}_expr", [f"{disp} - {motion}"])
        integral_expr.add_deps(motion)
    fsbc = FSCouplingProblem(f"P{prefix}", space, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(lm, FSExpr(integral_expr, basis))
    fsbc.add_term(disp, FSExpr(lm, basis))
    fsbc.add_expr_deps(basis, integral_expr)
    return fsbc


def create_cl_coupling_problems(
    tops: Mapping[int, ICheartTopology],
    lms: Mapping[int, IVariable],
    space: IVariable,
    disp: IVariable,
    motion: IVariable | None,
    cl_part: CLPartition,
    cl_top: CLBasisExpressions,
) -> Mapping[int, FSCouplingProblem]:
    res = {
        i: create_cl_coupling_problem(
            f"{s}LM",
            space,
            cl_top["p"][i],
            lms[i],
            disp,
            motion,
            tops[i],
            # [lms[n] for n in [i - 1, i + 1] if n in lms],
        )
        for i, s in cl_part.n_prefix.items()
    }
    return res
