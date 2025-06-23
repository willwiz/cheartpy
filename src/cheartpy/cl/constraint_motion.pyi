from typing import overload

from cheartpy.cheart.physics.fs_coupling.struct import FSCouplingProblem
from cheartpy.cheart.trait import IVariable

from .struct import CLTopology

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
    cl: CLTopology | None,
    space: IVariable,
    lm: None,
    disp: IVariable,
    *motion: IVariable | None,
    sfx: str = "CL",
) -> None: ...
@overload
def create_cl_motion_constraint_problem(
    cl: CLTopology,
    space: IVariable,
    lm: IVariable,
    disp: IVariable,
    *motion: IVariable | None,
    sfx: str = "CL",
) -> FSCouplingProblem: ...
