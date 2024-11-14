__all__ = ["create_rotation_constraint"]
from typing import Literal, Mapping, Sequence
from ..cheart_core.physics import FSCouplingProblem, FSExpr
from ..cheart_core.interface import IVariable, ICheartTopology, IExpression
from ..cheart_core.implementation import Expression

ROT_CONS_CHOICE = Mapping[Literal["T", "R"], Sequence[Literal["x", "y", "z"]]]


def create_rotation_operator_expr(
    name: str,
    space: IVariable | IExpression,
    choice: ROT_CONS_CHOICE,
) -> Mapping[Literal["p", "m"], IExpression]:
    total_dof = sum(len(v) for v in choice.values())
    ROT_DOF: Mapping[
        Literal["T", "R"],
        Mapping[Literal["x", "y", "z"], Sequence[str | float]],
    ] = {
        "T": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
        "R": {
            "x": [0, f"{space}.3", f"-{space}.2"],
            "y": [f"-{space}.3", 0, f"{space}.1"],
            "z": [f"{space}.2", f"-{space}.1", 0],
        },
    }
    dof = [j for k, v in choice.items() for i in v for j in ROT_DOF[k][i]]
    if "R" not in choice:
        return {"p": Expression(name, dof), "m": Expression(name, dof)}
    if total_dof == 1:
        return {"p": Expression(name, dof), "m": Expression(name, dof)}
    return {
        "p": Expression(name, dof),
        "m": Expression(
            name + "_T", [dof[3 * i + j] for j in range(3) for i in range(total_dof)]
        ),
    }


def create_rotation_constraint(
    prefix: str,
    root: ICheartTopology,
    space: IVariable,
    disp: IVariable,
    lm: IVariable,
    choice: ROT_CONS_CHOICE,
):
    # rot_dof = Expression("RotMat", [0, f"{space}.3", f"-{space}.2"])
    rot_dof = create_rotation_operator_expr(f"{prefix}_matexpr", space, choice)
    rot_bc = FSCouplingProblem(f"{prefix}", space, root)
    rot_bc.set_lagrange_mult(lm, FSExpr(disp, rot_dof["p"]))
    rot_bc.add_term(disp, FSExpr(lm, rot_dof["m"]))
    rot_bc.add_expr_deps(*rot_dof.values())
    # rot_bc.add_aux_vars(root)
    return rot_bc
