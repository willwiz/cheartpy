from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict, Unpack, overload

from .aliases import (
    BoundaryType,
    CheartBasisType,
    CheartElementType,
    MatrixSolverOption,
    SolverSubgroupMethod,
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
    EXPRESSION_VALUE,
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

class _CreateBasisKwargs(TypedDict, total=False):
    gp: int

def hash_tops(tops: list[ICheartTopology] | list[str]) -> str: ...
def create_time_scheme(
    name: str,
    start: int,
    stop: int,
    step: float | str | Path,
) -> ITimeScheme: ...
def create_basis(
    elem: CheartElementType,
    kind: CheartBasisType,
    order: Literal[0, 1, 2],
    **kwargs: Unpack[_CreateBasisKwargs],
) -> ICheartBasis: ...
def create_boundary_basis(vol: ICheartBasis) -> ICheartBasis: ...
@overload
def create_topology(
    name: str,
    basis: ICheartBasis,
    mesh: Path | str,
    format: VariableExportFormat = ...,
) -> CheartTopology: ...
@overload
def create_topology(
    name: str,
    basis: None,
    mesh: Path | str,
    format: VariableExportFormat = ...,
) -> NullTopology: ...
def create_embedded_topology(
    name: str,
    embedded_top: CheartTopology,
    mesh: Path | str,
    fmt: VariableExportFormat = ...,
) -> CheartTopology: ...
def create_solver_matrix(
    name: str,
    solver: MatrixSolverOption,
    *probs: IProblem | None,
) -> ISolverMatrix: ...
def create_solver_group(
    name: str,
    time: ITimeScheme,
    *solver_subgroup: ISolverSubGroup,
) -> ISolverGroup: ...
def create_solver_subgroup(
    method: SolverSubgroupMethod,
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
    interface_file: Path | str,
    nest_in_boundary: int | None = None,
) -> ManyToOneTopInterface: ...
def create_bcpatch(
    label: int,
    var: IVariable | tuple[IVariable, int | None],
    kind: BoundaryType,
    *val: BC_VALUE,
) -> IBCPatch: ...
def create_bc(*val: IBCPatch) -> IBoundaryCondition: ...
def create_variable(
    name: str,
    top: ICheartTopology | None,
    dim: int = 3,
    data: Path | str | None = None,
    fmt: VariableExportFormat = ...,
    freq: int = 1,
    loop_step: int | None = None,
) -> IVariable: ...
def create_expr(name: str, value: Sequence[EXPRESSION_VALUE]) -> IExpression: ...
def add_statevar(p: IProblem | None, *var: IVariable | IExpression | None) -> None: ...
