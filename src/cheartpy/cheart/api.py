from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

__all__ = [
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
    Basis,
    BCPatch,
    BoundaryCondition,
    CheartBasis,
    CheartTopology,
    Expression,
    ManyToOneTopInterface,
    NullTopology,
    OneToOneTopInterface,
    Quadrature,
    SolverGroup,
    SolverMatrix,
    SolverSubGroup,
    TimeScheme,
    Variable,
)
from .string_tools import get_enum

if TYPE_CHECKING:
    from collections.abc import Sequence

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
        ITopInterface,
        IVariable,
    )


def hash_tops(tops: list[ICheartTopology] | list[str]) -> str:
    names = [str(t) for t in tops]
    return "_".join(names)


def create_time_scheme(
    name: str,
    start: int,
    stop: int,
    step: float | str,
) -> ITimeScheme:
    if isinstance(step, str) and not Path(step).is_file():
        msg = f"Time step file {step} is not found!"
        raise ValueError(msg)
    return TimeScheme(name, start, stop, step)


_ORDER = {
    0: "Z",
    1: "L",
    2: "Q",
    3: "C",
    4: "A",
    5: "U",
}

_ELEM = {
    CheartElementType.POINT_ELEMENT: "Point",
    CheartElementType.point: "Point",
    CheartElementType.ONED_ELEMENT: "Line",
    CheartElementType.line: "Line",
    CheartElementType.TRIANGLE_ELEMENT: "Tri",
    CheartElementType.tri: "Tri",
    CheartElementType.QUADRILATERAL_ELEMENT: "Quad",
    CheartElementType.quad: "Quad",
    CheartElementType.TETRAHEDRAL_ELEMENT: "Tet",
    CheartElementType.tet: "Tet",
    CheartElementType.HEXAHEDRAL_ELEMENT: "Hex",
    CheartElementType.hex: "Hex",
}


def create_basis(
    elem: CHEART_ELEMENT_TYPE | CheartElementType,
    kind: CHEART_BASES_TYPE | CheartBasisType,
    quadrature: CHEART_QUADRATURE_TYPE | CheartQuadratureType,
    order: Literal[0, 1, 2],
    gp: int,
) -> ICheartBasis:
    elem = get_enum(elem, CheartElementType)
    kind = get_enum(kind, CheartBasisType)
    quadrature = get_enum(quadrature, CheartQuadratureType)
    name = f"{_ORDER[order]}{_ELEM[elem]}"
    if 2 * gp < order + 1:
        msg = f"For {name}, order {2 * gp} < {order + 1}"
        raise ValueError(msg)
    if quadrature is CheartQuadratureType.KEAST_LYNESS and elem not in [
        CheartElementType.TETRAHEDRAL_ELEMENT,
        CheartElementType.TRIANGLE_ELEMENT,
    ]:
        msg = f"For {name} Basis, KEAST_LYNESS can only be used with tetrahedral or triangles"
        raise ValueError(msg)
    return CheartBasis(name, elem, Basis(kind, order), Quadrature(quadrature, gp))


def create_boundary_basis(vol: ICheartBasis) -> ICheartBasis:
    match vol.elem:
        case CheartElementType.HEXAHEDRAL_ELEMENT | CheartElementType.hex:
            elem = CheartElementType.QUADRILATERAL_ELEMENT
        case CheartElementType.TETRAHEDRAL_ELEMENT | CheartElementType.tet:
            elem = CheartElementType.TRIANGLE_ELEMENT
        case CheartElementType.QUADRILATERAL_ELEMENT | CheartElementType.quad:
            elem = CheartElementType.ONED_ELEMENT
        case CheartElementType.TRIANGLE_ELEMENT | CheartElementType.tri:
            elem = CheartElementType.ONED_ELEMENT
        case CheartElementType.ONED_ELEMENT | CheartElementType.line:
            elem = CheartElementType.POINT_ELEMENT
        case CheartElementType.POINT_ELEMENT | CheartElementType.point:
            msg = "No such thing as boundary for point elements"
            raise ValueError(msg)
    return CheartBasis(f"{vol}_surf", elem, vol.basis, vol.quadrature)


def create_topology(
    name: str,
    basis: ICheartBasis | None,
    mesh: str,
    fmt: VARIABLE_EXPORT_FORMAT | VariableExportFormat = VariableExportFormat.TXT,
) -> ICheartTopology:
    if basis is None:
        return NullTopology()
    fmt = get_enum(fmt, VariableExportFormat)
    return CheartTopology(name, basis, mesh, fmt)


def create_embedded_topology(
    name: str,
    embedded_top: ICheartTopology,
    mesh: str,
    fmt: VARIABLE_EXPORT_FORMAT | VariableExportFormat = VariableExportFormat.TXT,
) -> ICheartTopology:
    fmt = get_enum(fmt, VariableExportFormat)
    return CheartTopology(name, None, mesh, fmt, embedded=embedded_top)


def create_solver_matrix(
    name: str,
    solver: MATRIX_SOLVER_TYPES | MatrixSolverTypes,
    *probs: IProblem | None,
) -> ISolverMatrix:
    problems: dict[str, IProblem] = {}
    for p in probs:
        if p is not None:
            problems[str(p)] = p
    method = get_enum(solver, MatrixSolverTypes)
    return SolverMatrix(name, method, problems)


def create_solver_group(
    name: str,
    time: ITimeScheme,
    *solver_subgroup: ISolverSubGroup,
) -> ISolverGroup:
    return SolverGroup(name, time, list(solver_subgroup))


def create_solver_subgroup(
    method: SOLVER_SUBGROUP_ALGORITHM | SolverSubgroupAlgorithm,
    *probs: ISolverMatrix | IProblem,
) -> ISolverSubGroup:
    problems: dict[str, ISolverMatrix | IProblem] = {}
    for p in probs:
        problems[str(p)] = p
    return SolverSubGroup(get_enum(method, SolverSubgroupAlgorithm), problems)


def create_top_interface(
    method: Literal["OneToOne", "ManyToOne"],
    topologies: list[ICheartTopology],
    master_topology: ICheartTopology | None = None,
    interface_file: str | None = None,
    nest_in_boundary: int | None = None,
) -> ITopInterface:
    match method:
        case "OneToOne":
            name = hash_tops(topologies)
            return OneToOneTopInterface(name, topologies)
        case "ManyToOne":
            if master_topology is None:
                msg = "ManyToOne requires a master_topology"
                raise ValueError(msg)
            if interface_file is None:
                msg = "ManyToOne requires a interface_file"
                raise ValueError(msg)
            name = hash_tops(topologies) + ":" + str(master_topology)
            return ManyToOneTopInterface(
                name,
                topologies,
                master_topology,
                interface_file,
                nest_in_boundary,
            )


def create_bcpatch(
    label: int,
    var: IVariable | tuple[IVariable, int | None],
    kind: BOUNDARY_TYPE | BoundaryType,
    *val: BC_VALUE,
) -> IBCPatch:
    return BCPatch(label, var, get_enum(kind, BoundaryType), *val)


def create_bc(*val: IBCPatch) -> IBoundaryCondition:
    if len(val) > 0:
        return BoundaryCondition(val)
    return BoundaryCondition()


class _ExtraCreateVarOptions(TypedDict, total=False):
    fmt: VARIABLE_EXPORT_FORMAT | VariableExportFormat
    freq: int
    loop_step: int | None


def create_variable(
    name: str,
    top: ICheartTopology | None,
    dim: int = 3,
    data: str | None = None,
    **kwargs: Unpack[_ExtraCreateVarOptions],
) -> IVariable:
    fmt = get_enum(kwargs.get("fmt", VariableExportFormat.TXT), VariableExportFormat)
    top = NullTopology() if top is None else top
    return Variable(name, top, dim, data, fmt, kwargs.get("freq", 1), kwargs.get("loop_step"))


def create_expr(name: str, value: Sequence[EXPRESSION_VALUE]) -> IExpression:
    return Expression(name, value)


def add_statevar(p: IProblem | None, *var: IVariable | IExpression | None) -> None:
    if p is None:
        return
    for v in var:
        p.add_state_variable(v)
