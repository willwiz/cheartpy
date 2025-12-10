from typing import overload

from cheartpy.fe.physics.fs_coupling.struct import FSCouplingProblem
from cheartpy.fe.trait import IVariable

from .struct import CLStructure

@overload
def create_cl_motion_constraint_problem(
    cl: None,
    space: IVariable,
    lm: IVariable | None,
    disp: IVariable,
    *motion: IVariable | None,
    sfx: str = "CL",
) -> None: ...
@overload
def create_cl_motion_constraint_problem(
    cl: CLStructure | None,
    space: IVariable,
    lm: None,
    disp: IVariable,
    *motion: IVariable | None,
    sfx: str = "CL",
) -> None: ...
@overload
def create_cl_motion_constraint_problem(
    cl: CLStructure,
    space: IVariable,
    lm: IVariable,
    disp: IVariable,
    *motion: IVariable | None,
    sfx: str = "CL",
) -> FSCouplingProblem: ...
