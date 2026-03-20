from typing import TYPE_CHECKING, TypedDict, Unpack

from cheartpy.fe.aliases import (
    SolidProblemEnum,
    SolidProblemType,
)
from cheartpy.fe.physics.solid_mechanics.solid_problems import SolidProblem
from cheartpy.fe.utils import get_enum

if TYPE_CHECKING:
    from cheartpy.fe.trait import ILaw, IVariable


class _SolidProblemExtraArgs(TypedDict, total=False):
    vel: IVariable | None
    pres: IVariable | None
    matlaws: list[ILaw] | None


def create_solid_mechanics_problem(
    name: str,
    prob: SolidProblemType | SolidProblemEnum,
    space: IVariable,
    disp: IVariable,
    **kwargs: Unpack[_SolidProblemExtraArgs],
) -> SolidProblem:
    problem = get_enum(prob, SolidProblemEnum)
    if space.get_data() is None:
        msg = f"Space for {name} must be initialized with values"
        raise ValueError(msg)
    vel = kwargs.get("vel")
    if (problem, vel) == (SolidProblemEnum.TRANSIENT, None):
        msg = f"Solid Problem {name}: Transient must have Vel"
        raise ValueError(msg)
    return SolidProblem(
        name,
        problem,
        space,
        disp,
        vel=vel,
        pres=kwargs.get("pres"),
        matlaws=kwargs.get("matlaws"),
    )
