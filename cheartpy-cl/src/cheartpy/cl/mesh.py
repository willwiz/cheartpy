from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, TypedDict, Unpack

import numpy as np
from cheartpy.mesh.struct import (
    CheartMesh,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from cheartpy.mesh.surface_core.normals import (
    compute_mesh_outer_normal_at_nodes,
    compute_normal_surface_at_center,
)
from cheartpy.vtk.api import get_vtk_elem
from pytools.logging import NLOGGER
from pytools.result import Err, Ok, all_ok

from .struct import CLNodalData, CLPartition, PatchNode2ElemMap

if TYPE_CHECKING:
    from pytools.arrays import A1, A2, DType
    from pytools.logging._trait import ILogger

__all__ = [
    "create_cheart_cl_nodal_meshes",
    "create_cl_partition",
    "filter_mesh_normals",
]

_1_SQRT2 = 1.0 / np.sqrt(2.0)


def check_normal[F: np.floating](
    node_normals: A2[F],
    elem: A1[np.integer],
    patch_normals: A1[F],
    log: ILogger = NLOGGER,
) -> bool:
    check = all(abs(node_normals[i] @ patch_normals) > _1_SQRT2 for i in elem)
    if not check:
        log.debug(
            f"Normal check failed for elem = {elem}, patch normals of:",
            patch_normals,
            pformat([node_normals[i] for i in elem]),
        )
    return check


def filter_mesh_normals[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    elems: A2[I],
    normal_check: A2[F] | None,
    log: ILogger = NLOGGER,
) -> Ok[A2[I]] | Err:
    if normal_check is None:
        return Ok(elems)
    top_body_elem = get_vtk_elem(mesh.top.TYPE)
    if top_body_elem.surf is None:
        msg = "Attempting to compute normal from a 1D mesh, not possible"
        return Err(ValueError(msg))
    surf_type = get_vtk_elem(top_body_elem.surf)
    normals = compute_normal_surface_at_center(surf_type, mesh.space.v, elems, log)
    new_elems = np.array(
        [i for i, v in zip(elems, normals, strict=False) if check_normal(normal_check, i, v, log)],
        dtype=int,
    )
    return Ok(new_elems)


class _CreateCLPartKwargs(TypedDict, total=False):
    log: ILogger


def create_cl_partition[F: np.floating, I: np.integer](
    surf: tuple[str, int],
    ne: int,
    bc: tuple[float, float] = (0.0, 1.0),
    *,
    ftype: DType[F] = np.float64,
    dtype: DType[I] = np.intc,
    **kwargs: Unpack[_CreateCLPartKwargs],
) -> CLPartition[F, I]:
    log = kwargs.get("log", NLOGGER)
    prefix, in_surf = surf
    ftype = kwargs.get("ftype", np.float64)
    dtype = kwargs.get("dtype", np.intc)
    nn = ne + 1
    nodes = np.linspace(*bc, nn, dtype=ftype)
    elems = np.array([[i, i + 1] for i in range(ne)], dtype=dtype)
    support = np.zeros((nn, 3), dtype=ftype)
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
    return CLPartition(prefix, in_surf, nn, ne, node_prefix, elem_prefix, nodes, elems, support)


def create_boundarynode_map[F: np.floating, I: np.integer](
    cl: A2[F],
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
    domain: tuple[F, F] | A1[F],
) -> A1[np.intc]:
    nodes = node_map.i[
        ((node_map.x - domain[0]) > _OPTIMIZATION_TOL)
        & (domain[1] - node_map.x > _OPTIMIZATION_TOL)
    ]
    return np.fromiter({v for i in nodes for v in node_map.n2e_map[int(i)]}, dtype=np.intc)


class _CLNodalMeshKwargs[F: np.floating](TypedDict, total=False):
    normal_check: A2[F] | None
    log: ILogger


def create_cheartmesh_in_clrange[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    surf: CheartMeshPatch[I],
    bnd_map: PatchNode2ElemMap,
    domain: tuple[F, F] | A1[F],
    **kwargs: Unpack[_CLNodalMeshKwargs[F]],
) -> Ok[CheartMesh[F, I]] | Err:
    # Unpack the kwargs
    log = kwargs.get("log", NLOGGER)
    normal_check = kwargs.get("normal_check")
    # Main logic
    body_elem = get_vtk_elem(mesh.top.TYPE)
    if body_elem.surf is None:
        return Err(ValueError("Mesh is 1D, normal not defined"))
    log.debug(f"{domain=}")
    elems: A2[I] = surf.v[get_boundaryelems_in_clrange(bnd_map, domain)]
    log.debug(f"{len(elems)=}")
    log.debug(f"The number of elements in patch is {len(elems)}")
    match filter_mesh_normals(mesh, elems, normal_check, log):
        case Ok(elems):
            pass
        case Err(e):
            return Err(e)
    log.debug(f"The number of elements in patch normal filtering is {len(elems)}")
    nodes = np.unique(elems)
    node_map: Mapping[int, int] = {int(v): i for i, v in enumerate(nodes)}
    space = CheartMeshSpace(len(nodes), mesh.space.v[nodes])
    top = CheartMeshTopology(
        len(elems),
        np.array([[node_map[int(i)] for i in e] for e in elems], dtype=int),
        body_elem.surf,
    )
    return Ok(CheartMesh(space, top, None))


type NODAL_MESHES[F: np.floating, I: np.integer] = Mapping[int, CLNodalData[F, I]]


def create_cheart_cl_nodal_meshes[F: np.floating, I: np.integer](
    mesh_dir: Path | str,
    cheart_mesh: CheartMesh[F, I],
    cl: A2[F],
    cl_top: CLPartition[F, I],
    surf_id: int,
    **kwargs: Unpack[_CLNodalMeshKwargs[F]],
) -> Ok[NODAL_MESHES[F, I]] | Err:
    # Unpack the kwargs
    mesh_dir = Path(mesh_dir)
    log = kwargs.get("log", NLOGGER)
    normal_check = kwargs.get("normal_check")
    # Main logic
    if cheart_mesh.bnd is None:
        msg = "Mesh has not boundary"
        return Err(ValueError(msg))
    surf = cheart_mesh.bnd.v[surf_id]
    bnd_map = create_boundarynode_map(cl, surf)
    match all_ok(
        {
            k: create_cheartmesh_in_clrange(
                cheart_mesh,
                surf,
                bnd_map,
                (m, r),
                normal_check=normal_check,
                log=log,
            )
            for k, (m, _, r) in enumerate(cl_top.support)
        }
    ):
        case Ok(tops):
            nodal_meshes = {
                k: CLNodalData(
                    file=mesh_dir / v,
                    mesh=tops[k],
                    n=compute_mesh_outer_normal_at_nodes(tops[k], log),
                )
                for k, v in cl_top.n_prefix.items()
            }
        case Err(e):
            return Err(e)
    return Ok(nodal_meshes)


def assemble_linear_cl_mesh[F: np.floating, I: np.integer](
    nodal_meshes: NODAL_MESHES[F, I],
    node_offset: A1[I],
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
    itype = linear_mesh.top.v.dtype
    ftype = linear_mesh.space.v.dtype
    cl_0_x = np.ascontiguousarray(
        [linear_mesh.space.v[k].mean(axis=0) for k in linear_mesh.top.v],
        dtype=ftype,
    )
    cl_0_t = np.arange(0, linear_mesh.top.v.shape[0], dtype=itype).reshape(-1, 1)
    return CheartMesh(
        CheartMeshSpace(len(cl_0_x), cl_0_x),
        CheartMeshTopology(len(cl_0_t), cl_0_t, linear_mesh.top.TYPE),
        None,
    )


def assemble_interface_cl_mesh[F: np.floating, I: np.integer](
    cl_top: CLPartition[F, I],
    const_mesh: CheartMesh[F, I],
    node_count: A1[I],
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
    mesh_dir: Path | str,
    mesh: CheartMesh[F, I],
    cl: A2[F],
    cl_top: CLPartition[F, I],
    surf_id: int,
    **kwargs: Unpack[_CLNodalMeshKwargs[F]],
) -> Ok[tuple[CheartMesh[F, I], CheartMesh[F, I]]] | Err:
    # Unpack kwargs
    log = kwargs.get("log", NLOGGER)
    normal_check = kwargs.get("normal_check")
    # Main Logic
    match create_cheart_cl_nodal_meshes(
        mesh_dir,
        mesh,
        cl,
        cl_top,
        surf_id,
        normal_check=normal_check,
        log=log,
    ):
        case Ok(nodal_meshes):
            pass
        case Err(e):
            return Err(e)
    node_count = [len(x["mesh"].space.v) for x in nodal_meshes.values()]
    node_offset = np.add.accumulate(np.insert(node_count, 0, 0))
    linear_mesh = assemble_linear_cl_mesh(nodal_meshes, node_offset)
    # const_mesh = assemble_const_cl_mesh(linear_mesh)
    elem_count = np.array([x["mesh"].top.n for x in nodal_meshes.values()])
    interface_mesh = assemble_interface_cl_mesh(cl_top, linear_mesh, elem_count)
    return Ok((linear_mesh, interface_mesh))
