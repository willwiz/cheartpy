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
import os
from .aliases import *
from .interface import *
from .implementation import *
from .pytools import get_enum


def hash_tops(tops: list[ICheartTopology] | list[str]) -> str:
    names = [str(t) for t in tops]
    return "_".join(names)


def create_time_scheme(
    name: str, start: int, stop: int, step: float | str
) -> ITimeScheme:
    if isinstance(step, str):
        if not os.path.isfile(step):
            raise ValueError(f"Time step file {step} is not found!")
    return TimeScheme(name, start, stop, step)


def create_basis(
    name: str,
    elem: CHEART_ELEMENT_TYPE | CheartElementType,
    kind: CHEART_BASES_TYPE | CheartBasisType,
    quadrature: CHEART_QUADRATURE_TYPE | CheartQuadratureType,
    order: int,
    gp: int,
) -> ICheartBasis:
    elem = get_enum(elem, CheartElementType)
    kind = get_enum(kind, CheartBasisType)
    quadrature = get_enum(quadrature, CheartQuadratureType)
    if 2 * gp == order:
        raise ValueError(f"For {name}, order {order} <= {2 * gp - 1}")
    if quadrature is CheartQuadratureType.KEAST_LYNESS:
        if not elem in [
            CheartElementType.TETRAHEDRAL_ELEMENT,
            CheartElementType.TRIANGLE_ELEMENT,
        ]:
            raise ValueError(
                f"For {name} Basis, KEAST_LYNESS can only be used with tetrahydral or triangles"
            )
    return CheartBasis(name, elem, Basis(kind, order), Quadrature(quadrature, gp))


def create_topology(
    name: str,
    basis: ICheartBasis | None,
    mesh: str,
    format: VARIABLE_EXPORT_FORMAT | VariableExportFormat = VariableExportFormat.TXT,
) -> ICheartTopology:
    if basis is None:
        return NullTopology()
    fmt = get_enum(format, VariableExportFormat)
    return CheartTopology(name, basis, mesh, fmt)


def create_embedded_topology(
    name: str,
    embedded_top: ICheartTopology,
    mesh: str,
    format: VARIABLE_EXPORT_FORMAT | VariableExportFormat = VariableExportFormat.TXT,
) -> ICheartTopology:
    fmt = get_enum(format, VariableExportFormat)
    return CheartTopology(name, None, mesh, fmt, embedded=embedded_top)


def create_variable(
    name: str,
    top: ICheartTopology | None,
    dim: int = 3,
    data: str | None = None,
    format: VARIABLE_EXPORT_FORMAT | VariableExportFormat = VariableExportFormat.TXT,
    freq: int = 1,
    loop_step: int | None = None,
) -> IVariable:
    fmt = get_enum(format, VariableExportFormat)
    top = NullTopology() if top is None else top
    return Variable(name, top, dim, data, fmt, freq, loop_step)


def create_solver_matrix(
    name: str, solver: MATRIX_SOLVER_TYPES | MatrixSolverTypes, *probs: IProblem
) -> ISolverMatrix:
    problems: dict[str, IProblem] = dict()
    for p in probs:
        problems[str(p)] = p
    method = get_enum(solver, MatrixSolverTypes)
    return SolverMatrix(name, method, problems)


def create_solver_group(
    name: str, time: ITimeScheme, *solver_subgroup: ISolverSubGroup
) -> ISolverGroup:
    sub_group: list[ISolverSubGroup] = list()
    for sg in solver_subgroup:
        sub_group.append(sg)
    return SolverGroup(name, time, sub_group)


def create_solver_subgroup(
    method: SOLVER_SUBGROUP_ALGORITHM | SolverSubgroupAlgorithm,
    *probs: ISolverMatrix | IProblem,
) -> ISolverSubGroup:
    problems: dict[str, ISolverMatrix | IProblem] = dict()
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
                raise ValueError("ManyToOne requires a master_topology")
            if interface_file is None:
                raise ValueError("ManyToOne requires a interface_file")
            name = hash_tops(topologies) + ":" + str(master_topology)
            return ManyToOneTopInterface(
                name, topologies, master_topology, interface_file, nest_in_boundary
            )
