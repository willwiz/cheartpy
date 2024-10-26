import numpy as np
from typing import cast
from collections import defaultdict
from .types import *
from ..tools.path_tools import path
from ..cheart_core.interface.basis import _Variable, _Expression
from ..cheart_core.implementation.expressions import Expression
from ..meshing.cheart.data import *
from ..meshing.cheart.tools import compute_normal_surface
from ..meshing.cheart.elements import VtkType
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


def LL_interp(top: CLTopology, var: Mat[f64] | Vec[f64], cl: Vec[f64]) -> Mat[f64]:
    x_bar = [
        v for elem in top.elem for v in LL_basis(var[elem], top.node[elem], cl).values()
    ]
    return cast(Mat[f64], sum(x_bar))


def LL_expr(
    name: str, v: _Variable, b: tuple[float, float, float] | Vec[f64]
) -> _Expression:
    return Expression(
        name,
        [
            f"max(min(({v} {-b[0]:+.8g})/({b[1] - b[0]:.8g}),({b[2]:.8g} - {v})/({b[2] - b[1]:.8g})), 0)"
        ],
    )


def check_normal(node_normals: Mat[f64], elem: Vec[i32], patch_normals: Vec[f64]):
    return all([cast(bool, abs(node_normals[i] @ patch_normals) > 0.8) for i in elem])


def filter_mesh_normals(
    mesh: CheartMesh,
    elems: Mat[i32],
    normal_check: Mat[f64],
    log: BasicLogger | NullLogger = NullLogger(),
):
    log.debug(f"The number of elements in patch is {len(elems)}")
    normals = compute_normal_surface(VtkType.TriangleLinear, mesh.space.v, elems)
    elems = np.array(
        [i for i, v in zip(elems, normals) if check_normal(normal_check, i, v)],
        dtype=int,
    )
    log.debug(f"The number of elements in patch normal filtering is {len(elems)}")
    return elems


def create_cl_topology(
    ne: int,
    bc: tuple[float, float] = (0.0, 1.0),
    prefix: str = "CL",
    LOG: BasicLogger | NullLogger = NullLogger(),
) -> CLTopology:
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
    return CLTopology(nn, ne, node_prefix, elem_prefix, nodes, elems, support)


def create_clbasis_expr(
    var: _Variable,
    cl: CLTopology,
) -> CLBasisExpressions:
    pelem = {
        i: LL_expr(f"{s}B_p", var, cl.support[i]) for i, s in cl.node_prefix.items()
    }
    melem = {
        i: Expression(f"{s}B_m", [f"-{pelem[i]}"]) for i, s in cl.node_prefix.items()
    }
    first = next(iter(pelem.values()))
    first.add_var_deps(var)
    return {"pelem": pelem, "melem": melem}


def create_boundarynode_map(cl: Mat[f64], b: CheartMeshSurface) -> PatchNode2ElemMap:
    n2p_map: Mapping[int, list[int]] = defaultdict(list)
    for k, vs in enumerate(b.v):
        for v in vs:
            n2p_map[v].append(k)
    space_key = np.fromiter(n2p_map.keys(), dtype=int)
    return PatchNode2ElemMap(space_key, cl[space_key, 0], n2p_map)


def get_boundaryelems_in_clrange(
    map: PatchNode2ElemMap,
    domain: tuple[float, float] | Vec[f64],
) -> Vec[i32]:
    nodes = map.i[((map.x - domain[0]) > 1.0e-10) & (domain[1] - map.x > 1.0e-10)]
    elems = np.fromiter(set([v for i in nodes for v in map.n2e_map[i]]), dtype=int)
    return elems


def create_cheartmesh_in_clrange(
    mesh: CheartMesh,
    surf: CheartMeshSurface,
    bnd_map: PatchNode2ElemMap,
    domain: tuple[float, float] | Vec[f64],
    normal_check: Mat[f64] | None = None,
    LOG: BasicLogger | NullLogger = NullLogger(),
) -> CheartMesh:
    LOG.debug(f"{domain=}")
    elems = surf.v[get_boundaryelems_in_clrange(bnd_map, domain)]
    LOG.debug(f"{len(elems)=}")
    if normal_check is not None:
        elems = filter_mesh_normals(mesh, elems, normal_check, LOG)
    nodes = np.unique(elems)
    node_map: Mapping[int, int] = {v: i for i, v in enumerate(nodes)}
    space = CheartMeshSpace(len(nodes), mesh.space.v[nodes])
    top = CheartMeshTopology(
        len(elems), np.array([[node_map[i] for i in e] for e in elems], dtype=int)
    )
    return CheartMesh(space, top, None)


def create_cheart_cl_nodal_meshes(
    mesh_dir: str,
    mesh: CheartMesh,
    cl: Mat[f64],
    cl_top: CLTopology,
    inner_surf_id: int = 3,
    normal_check: Mat[f64] | None = None,
    LOG: BasicLogger | NullLogger = NullLogger(),
):
    if mesh.bnd is None:
        raise ValueError("Mesh has not boundary")
    surf = mesh.bnd.v[inner_surf_id]
    bnd_map = create_boundarynode_map(cl, surf)
    tops = {
        path(mesh_dir, cl_top.node_prefix[k]): create_cheartmesh_in_clrange(
            mesh, surf, bnd_map, (l, r), normal_check=normal_check, LOG=LOG
        )
        for k, (l, c, r) in enumerate(cl_top.support)
    }
    # top_names = [path(mesh_dir, cl_top.node_prefix[i]) for v in cl_top.]
    # for name, t in zip(top_names, tops):
    #     if not os.path.isfile(name):
    #         t.save(name)
    return tops
