__all__ = [
    "add_statevar",
    "create_basis",
    "create_bc",
    "create_bcpatch",
    "create_boundary_basis",
    "create_embedded_topology",
    "create_expr",
    "create_solver_group",
    "create_solver_matrix",
    "create_solver_subgroup",
    "create_time_scheme",
    "create_top_interface",
    "create_topology",
    "create_variable",
    "hash_tops",
]
from collections.abc import Sequence
from typing import Literal, overload

from cheartpy.cheart.trait.basic import EXPRESSION_VALUE

from .aliases import (
    BOUNDARY_TYPE,
    CHEART_BASES_TYPE,
    CHEART_ELEMENT_TYPE,
    CHEART_QUADRATURE_TYPE,
    MATRIX_SOLVER_TYPES,
    SOLVER_SUBGROUP_ALGORITHM,
    VARIABLE_EXPORT_FORMAT,
    BoundaryType,
    CheartBasisType,
    CheartElementType,
    CheartQuadratureType,
    MatrixSolverTypes,
    SolverSubgroupAlgorithm,
    VariableExportFormat,
)
from .impl import (
    CheartTopology,
    ManyToOneTopInterface,
    NullTopology,
    OneToOneTopInterface,
)
from .trait import (
    BC_VALUE,
    IBCPatch,
    IBoundaryCondition,
    ICheartBasis,
    ICheartTopology,
    IExpression,
    IProblem,
    ISolverGroup,
    ISolverMatrix,
    ISolverSubGroup,
    ITimeScheme,
    IVariable,
)

def hash_tops(tops: list[ICheartTopology] | list[str]) -> str: ...
def create_time_scheme(
    name: str,
    start: int,
    stop: int,
    step: float | str,
) -> ITimeScheme: ...
def create_basis(
    elem: CHEART_ELEMENT_TYPE | CheartElementType,
    kind: CHEART_BASES_TYPE | CheartBasisType,
    quadrature: CHEART_QUADRATURE_TYPE | CheartQuadratureType,
    order: Literal[0, 1, 2],
    gp: int,
) -> ICheartBasis: ...
def create_boundary_basis(vol: ICheartBasis) -> ICheartBasis: ...
@overload
def create_topology(
    name: str,
    basis: ICheartBasis,
    mesh: str,
    format: VARIABLE_EXPORT_FORMAT | VariableExportFormat = ...,
) -> CheartTopology: ...
@overload
def create_topology(
    name: str,
    basis: None,
    mesh: str,
    format: VARIABLE_EXPORT_FORMAT | VariableExportFormat = ...,
) -> NullTopology: ...
def create_embedded_topology(
    name: str,
    embedded_top: CheartTopology,
    mesh: str,
    fmt: VARIABLE_EXPORT_FORMAT | VariableExportFormat = ...,
) -> CheartTopology: ...
def create_solver_matrix(
    name: str,
    solver: MATRIX_SOLVER_TYPES | MatrixSolverTypes,
    *probs: IProblem | None,
) -> ISolverMatrix: ...
def create_solver_group(
    name: str,
    time: ITimeScheme,
    *solver_subgroup: ISolverSubGroup,
) -> ISolverGroup: ...
def create_solver_subgroup(
    method: SOLVER_SUBGROUP_ALGORITHM | SolverSubgroupAlgorithm,
    *probs: ISolverMatrix | IProblem,
) -> ISolverSubGroup: ...
@overload
def create_top_interface(
    method: Literal["OneToOne"],
    topologies: list[ICheartTopology],
    nest_in_boundary: int | None = None,
) -> OneToOneTopInterface: ...
@overload
def create_top_interface(
    method: Literal["ManyToOne"],
    topologies: list[ICheartTopology],
    master_topology: ICheartTopology,
    interface_file: str,
    nest_in_boundary: int | None = None,
) -> ManyToOneTopInterface: ...
def create_bcpatch(
    label: int,
    var: IVariable | tuple[IVariable, int | None],
    kind: BOUNDARY_TYPE | BoundaryType,
    *val: BC_VALUE,
) -> IBCPatch: ...
def create_bc(*val: IBCPatch) -> IBoundaryCondition: ...
def create_variable(
    name: str,
    top: ICheartTopology | None,
    dim: int = 3,
    data: str | None = None,
    fmt: VARIABLE_EXPORT_FORMAT | VariableExportFormat = ...,
    freq: int = 1,
    loop_step: int | None = None,
) -> IVariable: ...
def create_expr(name: str, value: Sequence[EXPRESSION_VALUE]) -> IExpression: ...
def add_statevar(p: IProblem | None, *var: IVariable | IExpression | None) -> None: ...
