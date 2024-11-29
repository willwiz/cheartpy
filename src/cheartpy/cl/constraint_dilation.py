__all__ = ["create_dilation_var_problem", "create_dilation_var_problems"]
from typing import Mapping
from ..cheart.physics import FSCouplingProblem, FSExpr
from ..cheart.trait import IVariable, ICheartTopology, IExpression
from ..cheart.impl.expressions import Expression
from .types import CLPartition, CLBasisExpressions


def create_dilation_var_problem(
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
    fsbc.set_lagrange_mult(lm, FSExpr(integral_expr, basis, "trace"))
    fsbc.add_deps(integral_expr)
    return fsbc


def create_dilation_var_problems(
    space: IVariable,
    disp: IVariable,
    motion: IVariable,
    cl_part: CLPartition,
    cl_basis: CLBasisExpressions,
    tops: Mapping[int, ICheartTopology],
    lms: Mapping[int, IVariable],
) -> dict[int, FSCouplingProblem]:
    res = {
        k: create_dilation_var_problem(
            f"{v}DL",
            space,
            cl_basis["p"][k],
            lms[k],
            disp,
            motion,
            tops[k],
        )
        for k, v in cl_part.n_prefix.items()
    }
    return res
