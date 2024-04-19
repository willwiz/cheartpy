__all__ = ["create_time_scheme", "create_basis", "create_topology", "create_variable"]
import os
from typing import Any, overload

from cheartpy.cheart_core.base_types.problems import _Problem
from cheartpy.cheart_core.solver_groups import SolverGroup, SolverSubGroup
from cheartpy.cheart_core.solver_matrices import SolverMatrix
from .base_types.variables import Variable
from .time_schemes import TimeScheme
from .base_types.topologies import _CheartTopology, NullTopology, CheartTopology
from .pytools import get_enum
from .aliases import *
from .base_types.basis import CheartBasis, Basis, Quadrature

def create_time_scheme(
    name: str, start: int, stop: int, step: float | str
) -> TimeScheme: ...
def create_basis(
    name: str,
    elem: CHEART_ELEMENT_TYPE | CheartElementType,
    kind: CHEART_BASES_TYPE | CheartBasisType,
    quadrature: CHEART_QUADRATURE_TYPE | CheartQuadratureType,
    order: int,
    gp: int,
) -> CheartBasis: ...
@overload
def create_topology(
    name: str,
    basis: CheartBasis,
    mesh: str,
    format: VARIABLE_EXPORT_FORMAT | VariableExportFormat = VariableExportFormat.TXT,
) -> CheartTopology: ...
@overload
def create_topology(
    name: str,
    basis: None,
    mesh: str,
    format: VARIABLE_EXPORT_FORMAT | VariableExportFormat = VariableExportFormat.TXT,
) -> NullTopology: ...
def create_embedded_topology(
    name: str,
    embedded_top: CheartTopology,
    mesh: str,
    format: VARIABLE_EXPORT_FORMAT | VariableExportFormat = VariableExportFormat.TXT,
) -> CheartTopology: ...
def create_variable(
    name: str,
    top: _CheartTopology | None,
    dim: int = 3,
    data: str | None = None,
    format: VARIABLE_EXPORT_FORMAT | VariableExportFormat = VariableExportFormat.TXT,
    freq: int = 1,
    loop_step: int | None = None,
) -> Variable: ...
def create_solver_matrix(
    name: str, solver: MATRIX_SOLVER_TYPES | MatrixSolverTypes, *probs: _Problem
) -> SolverMatrix: ...
def create_solver_group(
    name: str, time: TimeScheme, *solver_subgroup: SolverSubGroup
) -> SolverGroup: ...
def create_solver_subgroup(
    method: SOLVER_SUBGROUP_ALGORITHM | SolverSubgroupAlgorithm,
    *probs: SolverMatrix | _Problem,
) -> SolverSubGroup: ...
