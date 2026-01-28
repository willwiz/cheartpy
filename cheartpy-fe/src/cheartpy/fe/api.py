from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

from .aliases import (
    BoundaryEnum,
    BoundaryType,
    CheartBasisEnum,
    CheartBasisType,
    CheartElementEnum,
    CheartElementType,
    CheartQuadratureEnum,
    CheartTopInterfaceType,
    MatrixSolverEnum,
    MatrixSolverOption,
    SolverSubgroupMethod,
    SolverSubgroupMethodEnum,
    VariableExportEnum,
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
    CheartElementEnum.POINT_ELEMENT: "Point",
    CheartElementEnum.point: "Point",
    CheartElementEnum.ONED_ELEMENT: "Line",
    CheartElementEnum.line: "Line",
    CheartElementEnum.TRIANGLE_ELEMENT: "Tri",
    CheartElementEnum.tri: "Tri",
    CheartElementEnum.QUADRILATERAL_ELEMENT: "Quad",
    CheartElementEnum.quad: "Quad",
    CheartElementEnum.TETRAHEDRAL_ELEMENT: "Tet",
    CheartElementEnum.tet: "Tet",
    CheartElementEnum.HEXAHEDRAL_ELEMENT: "Hex",
    CheartElementEnum.hex: "Hex",
}

_QUADRATURE_FOR_ELEM: dict[CheartElementEnum, CheartQuadratureEnum] = {
    CheartElementEnum.POINT_ELEMENT: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.point: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.ONED_ELEMENT: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.line: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.TRIANGLE_ELEMENT: CheartQuadratureEnum.KEAST_LYNESS,
    CheartElementEnum.tri: CheartQuadratureEnum.KEAST_LYNESS,
    CheartElementEnum.QUADRILATERAL_ELEMENT: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.quad: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.TETRAHEDRAL_ELEMENT: CheartQuadratureEnum.KEAST_LYNESS,
    CheartElementEnum.tet: CheartQuadratureEnum.KEAST_LYNESS,
    CheartElementEnum.HEXAHEDRAL_ELEMENT: CheartQuadratureEnum.GAUSS_LEGENDRE,
    CheartElementEnum.hex: CheartQuadratureEnum.GAUSS_LEGENDRE,
}


class _CreateBasisKwargs(TypedDict, total=False):
    gp: int


def create_basis(
    elem: CheartElementType | CheartElementEnum,
    kind: CheartBasisType | CheartBasisEnum,
    order: Literal[0, 1, 2],
    **kwargs: Unpack[_CreateBasisKwargs],
) -> ICheartBasis:
    elem = get_enum(elem, CheartElementEnum)
    kind = get_enum(kind, CheartBasisEnum)
    quadrature = _QUADRATURE_FOR_ELEM[elem]
    name = f"{_ORDER[order]}{_ELEM[elem]}"
    gp = kwargs.get("gp", 9 if quadrature is CheartQuadratureEnum.GAUSS_LEGENDRE else 4)
    if 2 * gp < order + 1:
        msg = f"For {name}, order {2 * gp} < {order + 1}"
        raise ValueError(msg)
    if quadrature is CheartQuadratureEnum.KEAST_LYNESS and elem not in [
        CheartElementEnum.TETRAHEDRAL_ELEMENT,
        CheartElementEnum.TRIANGLE_ELEMENT,
    ]:
        msg = f"For {name} Basis, KEAST_LYNESS can only be used with tetrahedral or triangles"
        raise ValueError(msg)
    return CheartBasis(name, elem, Basis(kind, order), Quadrature(quadrature, gp))


def create_boundary_basis(vol: ICheartBasis) -> ICheartBasis:
    match vol.elem:
        case CheartElementEnum.HEXAHEDRAL_ELEMENT | CheartElementEnum.hex:
            elem = CheartElementEnum.QUADRILATERAL_ELEMENT
        case CheartElementEnum.TETRAHEDRAL_ELEMENT | CheartElementEnum.tet:
            elem = CheartElementEnum.TRIANGLE_ELEMENT
        case CheartElementEnum.QUADRILATERAL_ELEMENT | CheartElementEnum.quad:
            elem = CheartElementEnum.ONED_ELEMENT
        case CheartElementEnum.TRIANGLE_ELEMENT | CheartElementEnum.tri:
            elem = CheartElementEnum.ONED_ELEMENT
        case CheartElementEnum.ONED_ELEMENT | CheartElementEnum.line:
            elem = CheartElementEnum.POINT_ELEMENT
        case CheartElementEnum.POINT_ELEMENT | CheartElementEnum.point:
            msg = "No such thing as boundary for point elements"
            raise ValueError(msg)
    return create_basis(elem, vol.basis.kind, vol.basis.order, gp=vol.quadrature.gp)


def create_topology(
    name: str,
    basis: ICheartBasis | None,
    mesh: Path | str,
    fmt: VariableExportFormat | VariableExportEnum = VariableExportEnum.TXT,
) -> ICheartTopology:
    if basis is None:
        return NullTopology()
    fmt = get_enum(fmt, VariableExportEnum)
    return CheartTopology(name, basis, Path(mesh), fmt)


def create_embedded_topology(
    name: str,
    embedded_top: ICheartTopology,
    mesh: Path | str,
    fmt: VariableExportFormat | VariableExportEnum = VariableExportEnum.TXT,
) -> ICheartTopology:
    fmt = get_enum(fmt, VariableExportEnum)
    return CheartTopology(name, None, Path(mesh), fmt, embedded=embedded_top)


def create_solver_matrix(
    name: str,
    solver: MatrixSolverOption | MatrixSolverEnum,
    *probs: IProblem | None,
) -> ISolverMatrix:
    problems: dict[str, IProblem] = {}
    for p in probs:
        if p is not None:
            problems[str(p)] = p
    method = get_enum(solver, MatrixSolverEnum)
    return SolverMatrix(name, method, problems)


def create_solver_group(
    name: str,
    time: ITimeScheme,
    *solver_subgroup: ISolverSubGroup,
) -> ISolverGroup:
    return SolverGroup(name, time, list(solver_subgroup))


def create_solver_subgroup(
    method: SolverSubgroupMethod | SolverSubgroupMethodEnum,
    *probs: ISolverMatrix | IProblem,
) -> ISolverSubGroup:
    problems: dict[str, ISolverMatrix | IProblem] = {}
    for p in probs:
        problems[str(p)] = p
    return SolverSubGroup(get_enum(method, SolverSubgroupMethodEnum), problems)


def create_top_interface(
    method: CheartTopInterfaceType,
    topologies: list[ICheartTopology],
    master_topology: ICheartTopology | None = None,
    interface_file: Path | str | None = None,
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
                Path(interface_file),
                nest_in_boundary,
            )


def create_bcpatch(
    label: int,
    var: IVariable | tuple[IVariable, int | None],
    kind: BoundaryType | BoundaryEnum,
    *val: BC_VALUE,
) -> IBCPatch:
    return BCPatch(label, var, get_enum(kind, BoundaryEnum), *val)


def create_bc(*val: IBCPatch) -> IBoundaryCondition:
    if len(val) > 0:
        return BoundaryCondition(val)
    return BoundaryCondition()


class _ExtraCreateVarOptions(TypedDict, total=False):
    fmt: VariableExportFormat | VariableExportEnum
    freq: int
    loop_step: int | None


def create_variable(
    name: str,
    top: ICheartTopology | None,
    dim: int = 3,
    data: Path | str | None = None,
    **kwargs: Unpack[_ExtraCreateVarOptions],
) -> IVariable:
    fmt = get_enum(kwargs.get("fmt", VariableExportEnum.TXT), VariableExportEnum)
    top = NullTopology() if top is None else top
    data = Path(data) if data is not None else None
    return Variable(name, top, dim, data, fmt, kwargs.get("freq", 1), kwargs.get("loop_step"))


def create_expr(name: str, value: Sequence[EXPRESSION_VALUE]) -> IExpression:
    return Expression(name, value)


def add_statevar(p: IProblem | None, *var: IVariable | IExpression | None) -> None:
    if p is None:
        return
    for v in var:
        p.add_state_variable(v)
