__all__ = ["create_time_scheme", "create_basis",
           "create_topology", "create_variable"]
import os
from typing import Any, overload

from cheartpy.cheart_core.problems import Problem
from cheartpy.cheart_core.solver_groups import SolverGroup, SolverSubGroup
from cheartpy.cheart_core.solver_matrices import SolverMatrix
from .variables import Variable
from .time_schemes import TimeScheme
from .topologies import CheartTopology, NullTopology
from .pytools import get_enum
from .aliases import *
from .bases import CheartBasis, Basis, Quadrature


def create_time_scheme(name: str, start: int, stop: int, step: float | str) -> TimeScheme:
    if isinstance(step, str):
        if not os.path.isfile(step):
            raise ValueError(f"Time step file {step} is not found!")
    return TimeScheme(name, start, stop, step)


def create_basis(
    name: str, elem: CHEART_ELEMENT_TYPE | CheartElementType, kind: CHEART_BASES_TYPE | CheartBasisType,
    quadrature: CHEART_QUADRATURE_TYPE | CheartQuadratureType, order: int, gp: int,
) -> CheartBasis:
    elem = get_enum(elem, CheartElementType)
    kind = get_enum(kind, CheartBasisType)
    quadrature = get_enum(quadrature, CheartQuadratureType)
    if 2 * gp == order:
        raise ValueError(f"For {name}, order {order} <= {2 * gp - 1}")
    match quadrature, elem:
        case CheartQuadratureType.GAUSS_LEGENDRE, _:
            pass
        case CheartQuadratureType.KEAST_LYNESS, CheartElementType.TETRAHEDRAL_ELEMENT | CheartElementType.TRIANGLE_ELEMENT:
            pass
        case CheartQuadratureType.KEAST_LYNESS, _:
            raise ValueError(
                f"For {name} Basis, KEAST_LYNESS can only be used with tetrahydral or triangles")
    return CheartBasis(name, elem, Basis(kind, order), Quadrature(quadrature, gp))


@overload
def create_topology(
    name: str, basis: None, mesh: str, format: Any
) -> NullTopology: ...


@overload
def create_topology(
    name: str, basis: CheartBasis, mesh: str, format: VARIABLE_EXPORT_FORMAT | VariableExportFormat = VariableExportFormat.TXT
) -> CheartTopology: ...


def create_topology(
    name: str, basis: CheartBasis | None, mesh: str, format: VARIABLE_EXPORT_FORMAT | VariableExportFormat = VariableExportFormat.TXT
) -> CheartTopology | NullTopology:
    if basis is None:
        return NullTopology(name)
    else:
        for ext in ["_FE.X", "_FE.T", "_FE.B"]:
            file = mesh+ext
            if not os.path.isfile(file):
                raise ValueError(
                    f"Mesh file {file} not found for topology {name}")
    fmt = get_enum(format, VariableExportFormat)
    return CheartTopology(name, basis, mesh, fmt)


def create_variable(
    name: str, top: CheartTopology | None, dim: int = 3, space: str | None = None, format: VARIABLE_EXPORT_FORMAT | VariableExportFormat = VariableExportFormat.TXT,
    freq: int = 1, loop_step: int | None = None
) -> Variable:
    fmt = get_enum(format, VariableExportFormat)
    if space and top:
        if space != top.mesh:
            raise ValueError(f"Variable {
                             name} is a space variable but mesh file does not match {space} != {top.mesh}")
    return Variable(name, top, dim, space, fmt, freq, loop_step)


def create_solver_matrix(
    name: str, solver: MATRIX_SOLVER_TYPES | MatrixSolverTypes, *probs: Problem
) -> SolverMatrix:
    problems = dict()
    for p in probs:
        problems[p.name] = p
    method = get_enum(solver, MatrixSolverTypes)
    return SolverMatrix(name, method, problems)


def create_solver_group(
    name: str, time: TimeScheme, *solver_subgroup: SolverSubGroup
) -> SolverGroup:
    sub_group = list()
    for sg in solver_subgroup:
        sub_group.append(sg)
    return SolverGroup(name, time, sub_group)


def create_solver_subgroup(
    method: SOLVER_SUBGROUP_ALGORITHM | SolverSubgroupAlgorithm, *probs: SolverMatrix | Problem
) -> SolverSubGroup:
    problems = dict()
    for p in probs:
        problems[p.name] = p
    return SolverSubGroup(get_enum(method, SolverSubgroupAlgorithm), problems)
