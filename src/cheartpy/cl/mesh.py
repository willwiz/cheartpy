__all__ = [
    "filter_mesh_normals",
    "create_cl_partition",
    "create_cheart_cl_nodal_meshes",
]
import numpy as np
from typing import Mapping, cast
from collections import defaultdict
from ..tools.path_tools import path
from ..cheart_mesh import VTK_ELEM
from ..cheart_mesh.data import *
from ..mesh.surface_core import (
    compute_normal_surface_at_center,
    compute_mesh_outer_normal_at_nodes,
)
from ..var_types import *
from ..tools.basiclogging import *
from .data import *


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


def create_cl_partition(
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


NODAL_MESHES = Mapping[int, CLNodalData]


def create_cheart_cl_nodal_meshes(
    mesh_dir: str,
    mesh: CheartMesh,
    cl: Mat[f64],
    cl_top: CLPartition,
    surf_id: int,
    normal_check: Mat[f64] | None = None,
    LOG: ILogger = NullLogger(),
) -> NODAL_MESHES:
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


def assemble_linear_cl_mesh(
    nodal_meshes: NODAL_MESHES, node_offset: Vec[int_t]
) -> CheartMesh:
    cl_1_x = np.vstack([x["mesh"].space.v for x in nodal_meshes.values()], dtype=float)
    cl_1_t = np.vstack(
        [x["mesh"].top.v + i for x, i in zip(nodal_meshes.values(), node_offset)],
        dtype=int,
    )
    return CheartMesh(
        CheartMeshSpace(len(cl_1_x), cl_1_x),
        CheartMeshTopology(len(cl_1_t), cl_1_t, nodal_meshes[0]["mesh"].top.TYPE),
        None,
    )


def assemble_const_cl_mesh(linear_mesh: CheartMesh) -> CheartMesh:
    cl_0_x = np.ascontiguousarray(
        [linear_mesh.space.v[k].mean(axis=0) for k in linear_mesh.top.v], dtype=float
    )
    cl_0_t = np.arange(0, linear_mesh.top.v.shape[0], dtype=int).reshape(-1, 1)
    return CheartMesh(
        CheartMeshSpace(len(cl_0_x), cl_0_x),
        CheartMeshTopology(len(cl_0_t), cl_0_t, linear_mesh.top.TYPE),
        None,
    )


def assemble_interface_cl_mesh(
    cl_top: CLPartition, const_mesh: CheartMesh
) -> CheartMesh:
    cl_i_x = np.ascontiguousarray(
        [[c, 0, 0] for _, c, _ in cl_top.support], dtype=float
    )
    return CheartMesh(CheartMeshSpace(len(cl_i_x), cl_i_x), const_mesh.top, None)


def create_cheart_cl_topology_meshes(
    mesh_dir: str,
    mesh: CheartMesh,
    cl: Mat[f64],
    cl_top: CLPartition,
    surf_id: int,
    normal_check: Mat[f64] | None = None,
    LOG: ILogger = NullLogger(),
):
    nodal_meshes = create_cheart_cl_nodal_meshes(
        mesh_dir, mesh, cl, cl_top, surf_id, normal_check, LOG
    )
    node_offset = (lambda x: np.add.accumulate(x) - x[0])(
        [len(x["mesh"].space.v) for x in nodal_meshes.values()]
    )
    linear_mesh = assemble_linear_cl_mesh(nodal_meshes, node_offset)
    const_mesh = assemble_const_cl_mesh(linear_mesh)
    interface_mesh = assemble_interface_cl_mesh(cl_top, const_mesh)
    return linear_mesh, const_mesh, interface_mesh


def create_cheart_cl_topology_data(cl_top: CLPartition):
    vals = np.ascontiguousarray(list(cl_top.n_prefix.keys()))
    return np.ascontiguousarray(
        [vals == i for i in cl_top.n_prefix.keys()], dtype=float
    )
