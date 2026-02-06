from typing import TYPE_CHECKING, Literal, overload

from cheartpy.fe.aliases import MatrixSolverEnum, MatrixSolverOption
from cheartpy.fe.impl import MumpsMatrix, SolverMatrix
from cheartpy.fe.string_tools import get_enum

if TYPE_CHECKING:
    from cheartpy.fe.trait import IProblem, ISolverMatrix


@overload
def create_solver_matrix(
    name: str, solver: Literal["SOLVER_MUMPS"], *probs: IProblem | None
) -> MumpsMatrix: ...
@overload
def create_solver_matrix(
    name: str, solver: Literal["SOLVER_PETSC"], *probs: IProblem | None
) -> ISolverMatrix: ...
def create_solver_matrix(
    name: str,
    solver: MatrixSolverOption,
    *probs: IProblem | None,
) -> ISolverMatrix:
    problems: dict[str, IProblem] = {}
    for p in probs:
        if p is not None:
            problems[str(p)] = p
    method = get_enum(solver, MatrixSolverEnum)
    match solver:
        case "SOLVER_MUMPS":
            return MumpsMatrix(name, method, problems)
        case "SOLVER_PETSC":
            return SolverMatrix(name, method, problems)
