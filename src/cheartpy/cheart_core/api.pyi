__all__ = [
    "hash_tops",
    "create_time_scheme",
    "create_basis",
    "create_topology",
    "create_embedded_topology",
    "create_variable",
    "create_solver_matrix",
    "create_solver_subgroup",
    "create_solver_group",
    "create_top_interface",
]
from typing import overload
from .aliases import *
from .interface import *
from .implementation.topologies import (
    ManyToOneTopInterface,
    NullTopology,
    CheartTopology,
    OneToOneTopInterface,
)

def hash_tops(tops: list[_CheartTopology] | list[str]) -> str: ...
def create_time_scheme(
    name: str, start: int, stop: int, step: float | str
) -> _TimeScheme: ...
def create_basis(
    name: str,
    elem: CHEART_ELEMENT_TYPE | CheartElementType,
    kind: CHEART_BASES_TYPE | CheartBasisType,
    quadrature: CHEART_QUADRATURE_TYPE | CheartQuadratureType,
    order: int,
    gp: int,
) -> _CheartBasis: ...
@overload
def create_topology(
    name: str,
    basis: _CheartBasis,
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
) -> _Variable: ...
def create_solver_matrix(
    name: str, solver: MATRIX_SOLVER_TYPES | MatrixSolverTypes, *probs: _Problem
) -> _SolverMatrix: ...
def create_solver_group(
    name: str, time: _TimeScheme, *solver_subgroup: _SolverSubGroup
) -> _SolverGroup: ...
def create_solver_subgroup(
    method: SOLVER_SUBGROUP_ALGORITHM | SolverSubgroupAlgorithm,
    *probs: _SolverMatrix | _Problem,
) -> _SolverSubGroup: ...
@overload
def create_top_interface(
    method: Literal["OneToOne"],
    topologies: list[_CheartTopology],
    nest_in_boundary: int | None = None,
) -> OneToOneTopInterface: ...
@overload
def create_top_interface(
    method: Literal["ManyToOne"],
    topologies: list[_CheartTopology],
    master_topology: _CheartTopology,
    interface_file: str,
    nest_in_boundary: int | None = None,
) -> ManyToOneTopInterface: ...
