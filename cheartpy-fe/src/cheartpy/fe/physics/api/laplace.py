from typing import TYPE_CHECKING

from cheartpy.fe.physics.laplace import LaplaceProblem

if TYPE_CHECKING:
    from cheartpy.fe.trait import IVariable


def create_laplace_problem(name: str, space: IVariable, v: IVariable) -> LaplaceProblem:
    return LaplaceProblem(name, space, v)
