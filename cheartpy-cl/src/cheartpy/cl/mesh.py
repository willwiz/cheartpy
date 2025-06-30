from pathlib import Path
from pprint import pformat

from cheartpy.vtk.api import get_vtk_elem

__all__ = [
    "create_cheart_cl_nodal_meshes",
    "create_cl_partition",
    "filter_mesh_normals",
]
from collections import defaultdict
from collections.abc import Mapping
from typing import TypedDict, Unpack

import numpy as np
from arraystubs import Arr1, Arr2
from cheartpy.mesh.struct import (
    CheartMesh,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from cheartpy.mesh.surface_core.surface import (
    compute_mesh_outer_normal_at_nodes,
    compute_normal_surface_at_center,
)
from pytools.logging.api import NULL_LOGGER
from pytools.logging.trait import ILogger

from .struct import CLNodalData, CLPartition, PatchNode2ElemMap

_SQRT2 = np.sqrt(2.0)


def check_normal[F: np.floating](
    node_normals: Arr2[F],
    elem: Arr1[np.integer],
    patch_normals: Arr1[F],
    log: ILogger = NULL_LOGGER,
) -> bool:
    check = all(abs(node_normals[i] @ patch_normals) > _SQRT2 for i in elem)
    if not check:
        log.debug(
            f"Normal check failed for elem = {elem}, patch normals of:",
            patch_normals,
            pformat([node_normals[i] for i in elem]),
        )
    return check


def filter_mesh_normals[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    elems: Arr2[I],
    normal_check: Arr2[F],
    log: ILogger = NULL_LOGGER,
) -> Arr2[I]:
    log.debug(f"The number of elements in patch is {len(elems)}")
    top_body_elem = get_vtk_elem(mesh.top.TYPE)
    if top_body_elem.surf is None:
        msg = "Attempting to compute normal from a 1D mesh, not possible"
        raise ValueError(msg)
    surf_type = get_vtk_elem(top_body_elem.surf)
    normals = compute_normal_surface_at_center(surf_type, mesh.space.v, elems, log)
    elems = np.array(
        [i for i, v in zip(elems, normals, strict=False) if check_normal(normal_check, i, v, log)],
        dtype=int,
    )
    log.debug(f"The number of elements in patch normal filtering is {len(elems)}")
    return elems


def create_cl_partition(
    prefix: str,
    in_surf: int,
    ne: int,
    bc: tuple[float, float] = (0.0, 1.0),
    log: ILogger = NULL_LOGGER,
) -> CLPartition[np.float64, np.intc]:
    nn = ne + 1
    nodes = np.linspace(*bc, nn, dtype=float)
    elems = np.array([[i, i + 1] for i in range(ne)], dtype=int)
    support = np.zeros((nn, 3), dtype=float)
    support[0, 0] = nodes[0] - (nodes[1] - nodes[0])
    support[1:, 0] = nodes[:-1]
    support[:, 1] = nodes
    support[:-1, 2] = nodes[1:]
    support[-1, 2] = nodes[-1] + (nodes[-1] - nodes[-2])
    log.debug(f"{elems=}")
    log.debug(f"{nodes=}")
    log.debug(f"{support=}")
    node_prefix = dict(enumerate([f"{prefix}{k}" for k in range(nn)]))
    elem_prefix = dict(enumerate([f"{prefix}{k}E" for k in range(ne)]))
    log.debug(f"{node_prefix=}")
    log.debug(f"{elem_prefix=}")
    return CLPartition(
        prefix,
        in_surf,
        nn,
        ne,
        node_prefix,
        elem_prefix,
        nodes,
        elems,
        support,
    )


def create_boundarynode_map[F: np.floating, I: np.integer](
    cl: Arr2[F],
    b: CheartMeshPatch[I],
) -> PatchNode2ElemMap:
    n2p_map: Mapping[int, list[int]] = defaultdict(list)
    for k, vs in enumerate(b.v):
        for v in vs:
            n2p_map[int(v)].append(k)
    space_key = np.fromiter(n2p_map.keys(), dtype=np.intc)
    return PatchNode2ElemMap(space_key, cl[space_key, 0], n2p_map)


_OPTIMIZATION_TOL = 1.0e-8


def get_boundaryelems_in_clrange[F: np.floating](
    node_map: PatchNode2ElemMap,
    domain: tuple[F, F] | Arr1[F],
) -> Arr1[np.intc]:
    nodes = node_map.i[
        ((node_map.x - domain[0]) > _OPTIMIZATION_TOL)
        & (domain[1] - node_map.x > _OPTIMIZATION_TOL)
    ]
    return np.fromiter({v for i in nodes for v in node_map.n2e_map[int(i)]}, dtype=np.intc)


class _CLNodalMeshKwargs[F: np.floating](TypedDict, total=False):
    normal_check: Arr2[F] | None
    log: ILogger


def create_cheartmesh_in_clrange[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    surf: CheartMeshPatch[I],
    bnd_map: PatchNode2ElemMap,
    domain: tuple[F, F] | Arr1[F],
    **kwargs: Unpack[_CLNodalMeshKwargs[F]],
) -> CheartMesh[F, I]:
    # Unpack the kwargs
    log = kwargs.get("log", NULL_LOGGER)
    normal_check = kwargs.get("normal_check")
    # Main logic
    body_elem = get_vtk_elem(mesh.top.TYPE)
    if body_elem.surf is None:
        e = log.exception(ValueError("Mesh is 1D, normal not defined"))
        raise e
    log.debug(f"{domain=}")
    elems: Arr2[I] = surf.v[get_boundaryelems_in_clrange(bnd_map, domain)]
    log.debug(f"{len(elems)=}")
    if normal_check is not None:
        elems = filter_mesh_normals(mesh, elems, normal_check, log)
    nodes = np.unique(elems)
    node_map: Mapping[int, int] = {int(v): i for i, v in enumerate(nodes)}
    space = CheartMeshSpace(len(nodes), mesh.space.v[nodes])
    top = CheartMeshTopology(
        len(elems),
        np.array([[node_map[int(i)] for i in e] for e in elems], dtype=int),
        body_elem.surf,
    )
    return CheartMesh(space, top, None)


type NODAL_MESHES[F: np.floating, I: np.integer] = Mapping[int, CLNodalData[F, I]]


def create_cheart_cl_nodal_meshes[F: np.floating, I: np.integer](
    mesh_dir: str,
    mesh: CheartMesh[F, I],
    cl: Arr2[F],
    cl_top: CLPartition[F, I],
    surf_id: int,
    **kwargs: Unpack[_CLNodalMeshKwargs[F]],
) -> NODAL_MESHES[F, I]:
    # Unpack the kwargs
    log = kwargs.get("log", NULL_LOGGER)
    normal_check = kwargs.get("normal_check")
    # Main logic
    if mesh.bnd is None:
        msg = "Mesh has not boundary"
        raise ValueError(msg)
    surf = mesh.bnd.v[surf_id]
    bnd_map = create_boundarynode_map(cl, surf)
    tops = {
        k: create_cheartmesh_in_clrange(
            mesh,
            surf,
            bnd_map,
            (m, r),
            normal_check=normal_check,
            log=log,
        )
        for k, (m, _, r) in enumerate(cl_top.support)
    }
    log.debug("Computing mesh outer normals at every node.")
    return {
        k: CLNodalData(
            file=Path(mesh_dir, v),
            mesh=tops[k],
            n=compute_mesh_outer_normal_at_nodes(tops[k], log),
        )
        for k, v in cl_top.n_prefix.items()
    }


def assemble_linear_cl_mesh[F: np.floating, I: np.integer](
    nodal_meshes: NODAL_MESHES[F, I],
    node_offset: Arr1[I],
) -> CheartMesh[F, I]:
    cl_1_x = np.vstack([x["mesh"].space.v for x in nodal_meshes.values()], dtype=float)
    cl_1_t = np.vstack(
        [x["mesh"].top.v + i for x, i in zip(nodal_meshes.values(), node_offset, strict=False)],
        dtype=int,
    )
    return CheartMesh(
        CheartMeshSpace(len(cl_1_x), cl_1_x),
        CheartMeshTopology(len(cl_1_t), cl_1_t, nodal_meshes[0]["mesh"].top.TYPE),
        None,
    )


def assemble_const_cl_mesh[F: np.floating, I: np.integer](
    linear_mesh: CheartMesh[F, I],
) -> CheartMesh[F, I]:
    cl_0_x = np.ascontiguousarray(
        [linear_mesh.space.v[k].mean(axis=0) for k in linear_mesh.top.v],
        dtype=float,
    )
    cl_0_t = np.arange(0, linear_mesh.top.v.shape[0], dtype=int).reshape(-1, 1)
    return CheartMesh(
        CheartMeshSpace(len(cl_0_x), cl_0_x),
        CheartMeshTopology(len(cl_0_t), cl_0_t, linear_mesh.top.TYPE),
        None,
    )


def assemble_interface_cl_mesh[F: np.floating, I: np.integer](
    cl_top: CLPartition[F, I],
    const_mesh: CheartMesh[F, I],
    node_count: Arr1[I],
) -> CheartMesh[F, I]:
    cl_i_x = np.ascontiguousarray(
        [[c, 0, 0] for _, c, _ in cl_top.support],
        dtype=float,
    )
    cl_i_t = np.vstack(
        [np.full((x, 1), i) for i, x in enumerate(node_count)],
        dtype=int,
    )
    return CheartMesh(
        CheartMeshSpace(len(cl_i_x), cl_i_x),
        CheartMeshTopology(len(cl_i_t), cl_i_t, const_mesh.top.TYPE),
        None,
    )


def create_cheart_cl_topology_meshes[F: np.floating, I: np.integer](
    mesh_dir: str,
    mesh: CheartMesh[F, I],
    cl: Arr2[F],
    cl_top: CLPartition[F, I],
    surf_id: int,
    **kwargs: Unpack[_CLNodalMeshKwargs[F]],
) -> tuple[CheartMesh[F, I], CheartMesh[F, I]]:
    # Unpack kwargs
    log = kwargs.get("log", NULL_LOGGER)
    normal_check = kwargs.get("normal_check")
    # Main Logic
    nodal_meshes = create_cheart_cl_nodal_meshes(
        mesh_dir,
        mesh,
        cl,
        cl_top,
        surf_id,
        normal_check=normal_check,
        log=log,
    )
    node_count = [len(x["mesh"].space.v) for x in nodal_meshes.values()]
    node_offset = np.add.accumulate(np.insert(node_count, 0, 0))
    linear_mesh = assemble_linear_cl_mesh(nodal_meshes, node_offset)
    # const_mesh = assemble_const_cl_mesh(linear_mesh)
    elem_count = np.array([x["mesh"].top.n for x in nodal_meshes.values()])
    interface_mesh = assemble_interface_cl_mesh(cl_top, linear_mesh, elem_count)
    return linear_mesh, interface_mesh
