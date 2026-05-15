from cheartpy.fe.aliases import L2VarProjectionType
from cheartpy.fe.physics.l2_projection import L2VarProjection
from cheartpy.fe.trait import IVariable


def create_l2varprojection_problem(
    name: str,
    space: IVariable,
    rhs: IVariable,
    var: IVariable,
    projection: L2VarProjectionType = "gradient",
) -> L2VarProjection:
    return L2VarProjection(name, space, rhs, var, projection)
