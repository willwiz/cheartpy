from collections.abc import Collection, Mapping, Sequence
from typing import Literal, Required, TypedDict, Unpack

from cheartpy.fe.api import create_expr, create_variable
from cheartpy.fe.trait import ICheartTopology, IExpression, IVariable

from ._struct import FSCouplingProblem, FSExpr

__all__ = ["create_rotation_constraint"]


ROT_CONS_CHOICE = Mapping[Literal["T", "R"], Collection[Literal["x", "y", "z"]]]


def _u(vs: Sequence[IVariable | IExpression], i: int) -> str:
    if len(vs) == 1:
        return f"{vs[0]}.{i}"
    return f"({'-'.join(f'{v}.{i}' for v in vs)})"


def create_rotation_operator_expr(
    name: str,
    choice: ROT_CONS_CHOICE,
    *space: IVariable | IExpression,
) -> Mapping[Literal["p", "m"], IExpression]:
    total_dof = sum(len(v) for v in choice.values())
    rotational_dof: Mapping[
        Literal["T", "R"],
        Mapping[Literal["x", "y", "z"], Sequence[str | float]],
    ] = {
        "T": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
        "R": {
            "x": [0, f"{_u(space, 3)}", f"-{_u(space, 2)}"],
            "y": [f"-{_u(space, 3)}", 0, f"{_u(space, 1)}"],
            "z": [f"{_u(space, 2)}", f"-{_u(space, 1)}", 0],
        },
    }
    dof = [j for k, v in choice.items() for i in v for j in rotational_dof[k][i]]
    p_expr = create_expr(name, dof)
    p_expr.add_deps(*space)
    if total_dof == 1:
        return {"p": p_expr, "m": p_expr}
    m_expr = create_expr(
        name + "_T",
        [dof[3 * i + j] for j in range(3) for i in range(total_dof)],
    )
    return {"p": p_expr, "m": m_expr}


class _RotationConstraintVariable(TypedDict, total=False):
    space: Required[IVariable]
    disp: Required[IVariable | IExpression]
    domain: Sequence[IVariable | IExpression]
    freq: int


def create_rotation_constraint(
    prefix: str,
    root: ICheartTopology,
    choice: ROT_CONS_CHOICE,
    **kwargs: Unpack[_RotationConstraintVariable],
) -> FSCouplingProblem:
    """Create a rotation constraint for a given space and displacement variable."""
    rot_dof = sum(len(v) for v in choice.values())
    space, disp, freq = kwargs["space"], kwargs["disp"], kwargs.get("freq", -1)
    lm = create_variable(f"RLM{prefix}", None, rot_dof, freq=freq)
    rot_mat = create_rotation_operator_expr(
        f"PBRot{prefix}_Mexpr", choice, space, *kwargs.get("domain", ())
    )
    rot_bc = FSCouplingProblem(f"PRRot{prefix}", space, root)
    rot_bc.set_lagrange_mult(lm, FSExpr(disp, rot_mat["p"]))
    if isinstance(disp, IExpression):
        v_deps = disp.get_var_deps()
        if len(v_deps) != 1:
            msg = "Cannot deduce variable dependency for rotation constraint."
            raise ValueError(msg)
        disp_var = next(v_deps.__iter__())
    else:
        disp_var = disp
    rot_bc.add_term(disp_var, FSExpr(lm, rot_mat["m"]))
    rot_bc.add_expr_deps(*rot_mat.values())
    # rot_bc.add_aux_vars(root)
    return rot_bc
