from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict, Unpack, overload

from cheartpy.fe.aliases import (
    BoundaryType,
    CheartBasisType,
    CheartElementType,
    CheartQuadratureType,
    SolverSubgroupMethod,
    VariableExportFormat,
)
from cheartpy.fe.impl import CheartTopology, MumpsMatrix, NullTopology, PFile
from cheartpy.fe.trait import (
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
    ITopInterface,
    IVariable,
)

class _CreateBasisKwargs(TypedDict, total=False):
    quadrature: CheartQuadratureType
    gp: int

def create_basis(
    elem: CheartElementType, kind: CheartBasisType, order: int, **kwargs: Unpack[_CreateBasisKwargs]
) -> ICheartBasis: ...
def create_boundary_basis(vol: ICheartBasis) -> ICheartBasis: ...
def create_bcpatch(
    label: int,
    var: IVariable | tuple[IVariable, int | None],
    kind: BoundaryType,
    *val: BC_VALUE,
) -> IBCPatch: ...
def create_bc(*val: IBCPatch) -> IBoundaryCondition: ...
def create_expr(name: str, value: Sequence[EXPRESSION_VALUE]) -> IExpression: ...
@overload
def create_solver_matrix(
    name: str, solver: Literal["SOLVER_MUMPS"], *probs: IProblem | None
) -> MumpsMatrix: ...
@overload
def create_solver_matrix(
    name: str, solver: Literal["SOLVER_PETSC"], *probs: IProblem | None
) -> ISolverMatrix: ...
def create_pfile(header: str = "", output_dir: Path | None = None) -> PFile: ...
def create_solver_group(
    name: str,
    time: ITimeScheme,
    *solver_subgroup: ISolverSubGroup,
) -> ISolverGroup: ...
def create_solver_subgroup(
    method: SolverSubgroupMethod,
    *probs: ISolverMatrix | IProblem,
) -> ISolverSubGroup: ...
def create_time_scheme(
    name: str,
    start: int,
    stop: int,
    step: float | str | Path,
) -> ITimeScheme: ...
@overload
def create_topology(
    name: str, basis: ICheartBasis, mesh: Path | str, fmt: VariableExportFormat = ...
) -> CheartTopology: ...
@overload
def create_topology(
    name: str, basis: None, mesh: Path | str, fmt: VariableExportFormat = ...
) -> NullTopology: ...
def create_embedded_topology(
    name: str,
    embedded_top: ICheartTopology,
    mesh: Path | str,
    fmt: VariableExportFormat = "TXT",
) -> ICheartTopology: ...
@overload
def create_top_interface(
    method: Literal["OneToOne"],
    topologies: list[ICheartTopology],
    *,
    nest_in_bnd: int | None = None,
) -> ITopInterface: ...
@overload
def create_top_interface(
    method: Literal["ManyToOne"],
    topologies: list[ICheartTopology],
    *,
    master: ICheartTopology,
    interface_file: Path | str,
    nest_in_bnd: int | None = None,
) -> ITopInterface: ...
def add_statevar(p: IProblem | None, *var: IVariable | IExpression | None) -> None: ...

class _ExtraCreateVarOptions(TypedDict, total=False):
    fmt: VariableExportFormat
    freq: int
    loop_step: int | None

def create_variable(
    name: str,
    top: ICheartTopology | None,
    dim: int = 3,
    data: Path | str | None = None,
    **kwargs: Unpack[_ExtraCreateVarOptions],
) -> IVariable: ...
