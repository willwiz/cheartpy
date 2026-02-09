from typing import TYPE_CHECKING

from cheartpy.fe.aliases import SolverSubgroupMethod, SolverSubgroupMethodEnum
from cheartpy.fe.impl import SolverGroup, SolverSubGroup
from cheartpy.fe.utils import get_enum

if TYPE_CHECKING:
    from cheartpy.fe.trait import (
        IProblem,
        ISolverGroup,
        ISolverMatrix,
        ISolverSubGroup,
        ITimeScheme,
    )


def create_solver_group(
    name: str,
    time: ITimeScheme,
    *solver_subgroup: ISolverSubGroup,
) -> ISolverGroup:
    return SolverGroup(name, time, list(solver_subgroup))


def create_solver_subgroup(
    method: SolverSubgroupMethod,
    *probs: ISolverMatrix | IProblem,
) -> ISolverSubGroup:
    problems: dict[str, ISolverMatrix | IProblem] = {}
    for p in probs:
        problems[str(p)] = p
    return SolverSubGroup(get_enum(method, SolverSubgroupMethodEnum), problems)
