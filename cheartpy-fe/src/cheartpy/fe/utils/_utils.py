from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cheartpy.fe.trait import IExpression, IProblem, IVariable


def add_statevar(p: IProblem | None, *var: IVariable | IExpression | None) -> None:
    if p is None:
        return
    for v in var:
        p.add_state_variable(v)
