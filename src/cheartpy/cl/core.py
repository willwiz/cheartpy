__all__ = [
    "LL_interp",
    "L2norm",
    "filter_mesh_normals",
    "create_cl_topology",
    "create_clbasis_expr",
    "create_cheart_cl_nodal_meshes",
    "create_lms_on_cl",
]
import numpy as np
from typing import Mapping, TypedDict, cast
from collections import defaultdict
from .types import *
from ..tools.path_tools import path
from ..cheart.trait import IVariable, IExpression
from ..cheart.impl import Expression
from ..cheart.api import create_variable
from ..cheart_mesh import VTK_ELEM
from ..cheart_mesh.data import *
from ..mesh.surface_core.surface import (
    compute_normal_surface_at_center,
    compute_mesh_outer_normal_at_nodes,
)
from ..var_types import *
from ..tools.basiclogging import *


def L2norm(x: Vec[f64]) -> float:
    return cast(float, x @ x)


def LL_basis(
    var: Mat[f64] | Vec[f64], nodes: Vec[f64], x: Vec[f64]
) -> Mapping[int, Mat[f64]]:
    basis_func = {i: np.zeros_like(x) for i in range(2)}
    in_domain = (nodes[0] <= x) & (x <= nodes[1])
    basis_func[0][in_domain] = 1 - (x[in_domain] - nodes[0]) / (nodes[1] - nodes[0])
    basis_func[1][in_domain] = (x[in_domain] - nodes[0]) / (nodes[1] - nodes[0])
    return {k: var[k] * v[:, None] for k, v in basis_func.items()}


def LL_interp(top: CLPartition, var: Mat[f64] | Vec[f64], cl: Vec[f64]) -> Mat[f64]:
    x_bar = [
        v for elem in top.elem for v in LL_basis(var[elem], top.node[elem], cl).values()
    ]
    return cast(Mat[f64], sum(x_bar))


def LL_expr(
    name: str, v: IVariable, b: tuple[float, float, float] | Vec[f64]
) -> IExpression:
    basis = Expression(
        name,
        [
            f"max(min(({v}{-b[0]:+.8g})/({b[1] - b[0]:.8g}),({b[2]:.8g}-{v})/({b[2] - b[1]:.8g})), 0)"
        ],
    )
    basis.add_deps(v)
    return basis


def check_normal(
    node_normals: Mat[f64],
    elem: Vec[int_t],
    patch_normals: Vec[f64],
    LOG: ILogger = NullLogger(),
):
    check = all(
        [cast(bool, abs(node_normals[i] @ patch_normals) > 0.707) for i in elem]
    )
    if not check:
        LOG.debug(
            f"Normal check failed for elem = {elem}, normals of \n{[print(node_normals[i], patch_normals) for i in elem]}"
        )
    return check


def filter_mesh_normals(
    mesh: CheartMesh,
    elems: Mat[int_t],
    normal_check: Mat[f64],
    LOG: ILogger = NullLogger(),
):
    LOG.debug(f"The number of elements in patch is {len(elems)}")
    if mesh.top.TYPE.surf is None:
        raise ValueError(f"Attempting to compute normal from a 1D mesh, not possible")
    surf_type = VTK_ELEM[mesh.top.TYPE.surf]
    normals = compute_normal_surface_at_center(surf_type, mesh.space.v, elems, LOG)
    elems = np.array(
        [i for i, v in zip(elems, normals) if check_normal(normal_check, i, v, LOG)],
        dtype=int,
    )
    LOG.debug(f"The number of elements in patch normal filtering is {len(elems)}")
    return elems


def create_cl_topology(
    prefix: str,
    in_surf: int,
    ne: int,
    bc: tuple[float, float] = (0.0, 1.0),
    LOG: ILogger = NullLogger(),
) -> CLPartition:
    nn = ne + 1
    nodes = np.linspace(*bc, nn, dtype=float)
    elems = np.array([[i, i + 1] for i in range(ne)], dtype=int)
    support = np.zeros((nn, 3), dtype=float)
    support[0, 0] = nodes[0] - (nodes[1] - nodes[0])
    support[1:, 0] = nodes[:-1]
    support[:, 1] = nodes
    support[:-1, 2] = nodes[1:]
    support[-1, 2] = nodes[-1] + (nodes[-1] - nodes[-2])
    LOG.debug(f"{elems=}")
    LOG.debug(f"{nodes=}")
    LOG.debug(f"{support=}")
    node_prefix = {i: s for i, s in enumerate([f"{prefix}{k}" for k in range(nn)])}
    elem_prefix = {i: s for i, s in enumerate([f"{prefix}{k}E" for k in range(ne)])}
    LOG.debug(f"{node_prefix=}")
    LOG.debug(f"{elem_prefix=}")
    return CLPartition(
        prefix, in_surf, nn, ne, node_prefix, elem_prefix, nodes, elems, support
    )


def create_clbasis_expr(
    var: IVariable,
    cl: CLPartition,
) -> CLBasisExpressions:
    pelem = {i: LL_expr(f"{s}B_p", var, cl.support[i]) for i, s in cl.n_prefix.items()}
    melem = {i: Expression(f"{s}B_m", [f"-{pelem[i]}"]) for i, s in cl.n_prefix.items()}
    [m.add_deps(pelem[k]) for k, m in melem.items()]
    return {"p": pelem, "m": melem}


def create_boundarynode_map(cl: Mat[f64], b: CheartMeshPatch) -> PatchNode2ElemMap:
    n2p_map: Mapping[int, list[int]] = defaultdict(list)
    for k, vs in enumerate(b.v):
        for v in vs:
            n2p_map[v].append(k)
    space_key = np.fromiter(n2p_map.keys(), dtype=int)
    return PatchNode2ElemMap(space_key, cl[space_key, 0], n2p_map)


def get_boundaryelems_in_clrange(
    map: PatchNode2ElemMap,
    domain: tuple[float, float] | Vec[f64],
) -> Vec[int_t]:
    nodes = map.i[((map.x - domain[0]) > 1.0e-8) & (domain[1] - map.x > 1.0e-8)]
    elems = np.fromiter(set([v for i in nodes for v in map.n2e_map[i]]), dtype=int)
    return elems


def create_cheartmesh_in_clrange(
    mesh: CheartMesh,
    surf: CheartMeshPatch,
    bnd_map: PatchNode2ElemMap,
    domain: tuple[float, float] | Vec[f64],
    normal_check: Mat[f64] | None = None,
    LOG: ILogger = NullLogger(),
) -> CheartMesh:
    if mesh.top.TYPE.surf is None:
        e = LOG.exception(ValueError(f"Mesh is 1D, normal not defined"))
        raise e
    surf_type = VTK_ELEM[mesh.top.TYPE.surf]
    LOG.debug(f"{domain=}")
    elems = surf.v[get_boundaryelems_in_clrange(bnd_map, domain)]
    LOG.debug(f"{len(elems)=}")
    if normal_check is not None:
        elems = filter_mesh_normals(mesh, elems, normal_check, LOG)
    nodes = np.unique(elems)
    node_map: Mapping[int, int] = {v: i for i, v in enumerate(nodes)}
    space = CheartMeshSpace(len(nodes), mesh.space.v[nodes])
    top = CheartMeshTopology(
        len(elems),
        np.array([[node_map[i] for i in e] for e in elems], dtype=int),
        surf_type,
    )
    return CheartMesh(space, top, None)


class CLNodalData(TypedDict):
    file: str
    mesh: CheartMesh
    n: Mat[f64]


def create_cheart_cl_nodal_meshes(
    mesh_dir: str,
    mesh: CheartMesh,
    cl: Mat[f64],
    cl_top: CLPartition,
    surf_id: int,
    normal_check: Mat[f64] | None = None,
    LOG: ILogger = NullLogger(),
) -> Mapping[int, CLNodalData]:
    if mesh.bnd is None:
        raise ValueError("Mesh has not boundary")
    surf = mesh.bnd.v[surf_id]
    bnd_map = create_boundarynode_map(cl, surf)
    tops = {
        k: create_cheartmesh_in_clrange(
            mesh, surf, bnd_map, (l, r), normal_check=normal_check, LOG=LOG
        )
        for k, (l, _, r) in enumerate(cl_top.support)
    }
    LOG.debug(f"Computing mesh outer normals at every node.")
    return {
        k: {
            "file": path(mesh_dir, v),
            "mesh": tops[k],
            "n": compute_mesh_outer_normal_at_nodes(tops[k], LOG),
        }
        for k, v in cl_top.n_prefix.items()
    }


def create_lms_on_cl(
    prefix: str, cl: CLTopology | None, dim: int, ex_freq: int, set_bc: bool
) -> Mapping[int, IVariable]:
    if cl is None:
        return {}
    lms = {
        k: create_variable(f"{v.k}{prefix}", None, dim, freq=ex_freq)
        for k, v in cl.N.items()
    }
    if set_bc:
        keys = sorted(lms.keys())
        lms[keys[0]] = lms[keys[1]]
        lms[keys[-1]] = lms[keys[-2]]
    return lms
