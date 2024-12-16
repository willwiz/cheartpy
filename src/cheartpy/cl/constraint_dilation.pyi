__all__ = ["create_cl_dilation_constraint_problem"]
from typing import overload
from .data import CLTopology
from ..cheart.physics import FSCouplingProblem
from ..cheart.trait import IVariable

@overload
def create_cl_dilation_constraint_problem(
    cl: None,
    space: IVariable,
    lm: IVariable | None,
    disp: IVariable,
    *motion: IVariable | None,
    sfx: str = "DM",
) -> None: ...
@overload
def create_cl_dilation_constraint_problem(
    cl: CLTopology | None,
    space: IVariable,
    lm: None,
    disp: IVariable,
    *motion: IVariable | None,
    sfx: str = "DM",
) -> None: ...
@overload
def create_cl_dilation_constraint_problem(
    cl: CLTopology,
    space: IVariable,
    lm: IVariable,
    disp: IVariable,
    *motion: IVariable | None,
    sfx: str = "DM",
) -> FSCouplingProblem: ...
