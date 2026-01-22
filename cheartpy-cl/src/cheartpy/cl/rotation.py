from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

from cheartpy.fe.api import create_expr
from cheartpy.fe.physics.fs_coupling.struct import FSCouplingProblem, FSExpr

if TYPE_CHECKING:
    from cheartpy.fe.trait import ICheartTopology, IExpression, IVariable


__all__ = ["create_rotation_constraint"]


ROT_CONS_CHOICE = Mapping[Literal["T", "R"], Collection[Literal["x", "y", "z"]]]


def create_rotation_operator_expr(
    name: str,
    space: IVariable | IExpression,
    choice: ROT_CONS_CHOICE,
) -> Mapping[Literal["p", "m"], IExpression]:
    total_dof = sum(len(v) for v in choice.values())
    rotational_dof: Mapping[
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
    dof = [j for k, v in choice.items() for i in v for j in rotational_dof[k][i]]
    p_expr = create_expr(name, dof)
    if ("R" not in choice) or (total_dof == 1):
        return {"p": p_expr, "m": p_expr}
    m_expr = create_expr(
        name + "_T",
        [dof[3 * i + j] for j in range(3) for i in range(total_dof)],
    )
    return {"p": p_expr, "m": m_expr}


class _RotationConstraintVariable(TypedDict, total=True):
    space: IVariable
    disp: IVariable
    lm: IVariable


def create_rotation_constraint(
    prefix: str,
    root: ICheartTopology,
    choice: ROT_CONS_CHOICE,
    **kwargs: Unpack[_RotationConstraintVariable],
) -> FSCouplingProblem:
    """Create a rotation constraint for a given space and displacement variable."""
    # rot_dof = Expression("RotMat", [0, f"{space}.3", f"-{space}.2"])
    space, disp, lm = kwargs["space"], kwargs["disp"], kwargs["lm"]
    rot_dof = create_rotation_operator_expr(f"{prefix}_matexpr", space, choice)
    rot_bc = FSCouplingProblem(f"{prefix}", space, root)
    rot_bc.set_lagrange_mult(lm, FSExpr(disp, rot_dof["p"]))
    rot_bc.add_term(disp, FSExpr(lm, rot_dof["m"]))
    rot_bc.add_expr_deps(*rot_dof.values())
    # rot_bc.add_aux_vars(root)
    return rot_bc
