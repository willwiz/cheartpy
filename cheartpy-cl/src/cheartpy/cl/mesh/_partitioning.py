from typing import TYPE_CHECKING, TypedDict, Unpack, cast

import numpy as np
from cheartpy.mesh import CheartMesh, CheartMeshBoundary, CheartMeshSpace, CheartMeshTopology
from cheartpy.mesh_tools import create_index_permutation
from cheartpy.mesh_tools.tools import (
    ElemSearchMap,
    MergedMesh,
    build_element_searchmap,
    create_mesh_from_surface,
    merge_cheart_meshes,
)
from pytools.result import Err, Ok

from ._types import CLPartition

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A1
    from pytools.result import Result


def create_centerline_partition[F: np.floating = np.float64, I: np.integer = np.intp](
    param: int | A1[F] | CLPartition[F, I],
) -> CLPartition[F, I]:
    match param:
        case CLPartition():
            return param
        case int() as n:
            nodes = np.linspace(0, 1, n, dtype=np.float64)
        case np.ndarray() as nodes:
            ...
    elem_size = np.pad(np.diff(nodes), (1, 1), mode="edge")
    domain = np.stack(
        [nodes - elem_size[:-1] / 2, nodes + elem_size[1:] / 2], axis=1, dtype=nodes.dtype
    )
    top = np.hstack(
        (np.arange(len(nodes) - 1, dtype=np.intp), np.arange(1, len(nodes), dtype=np.intp))
    )
    return cast("CLPartition[F, I]", CLPartition(nodes=nodes, top=top, domain=domain))


class CLTopologyKwargs(TypedDict, total=False):
    search_map: ElemSearchMap


def create_mesh_for_cl_node[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    a_z: A1[np.floating],
    domain: A1[np.floating],
    **kwargs: Unpack[CLTopologyKwargs],
) -> CheartMesh[F, I]:
    search_map = kwargs.get("search_map") or build_element_searchmap(mesh.top.v).unwrap()
    index = np.flatnonzero((a_z >= domain[0]) & (a_z <= domain[1]))
    elements = np.unique([i for n in index for i in search_map[n]], sorted=True)
    connectivity = mesh.top.v[elements]
    perm = create_index_permutation(connectivity)
    new_x = CheartMeshSpace(mesh.space.v[perm.old])
    new_top = CheartMeshTopology(v=perm.fwd[connectivity], type=mesh.top.type)
    return CheartMesh(space=new_x, top=new_top, bnd=CheartMeshBoundary[I]({}))


def create_centerline_nodal_meshes[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    a_z: A1[np.floating],
    partition: int | A1[np.floating] | CLPartition[np.floating, np.integer],
    **kwargs: Unpack[CLTopologyKwargs],
) -> Mapping[int, CheartMesh[F, I]]:
    search_map = kwargs.get("search_map") or build_element_searchmap(mesh.top.v).unwrap()
    _partition = create_centerline_partition(partition)
    return {
        i: create_mesh_for_cl_node(mesh, a_z, domain, search_map=search_map)
        for i, domain in enumerate(_partition.domain)
    }


def create_centerline_mesh_in_volume[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    a_z: A1[np.floating],
    partition: int | A1[np.floating] | CLPartition[np.floating, np.integer],
    **kwargs: Unpack[CLTopologyKwargs],
) -> Result[MergedMesh[F, I]]:
    nodal_meshes = create_centerline_nodal_meshes(mesh, a_z, partition, **kwargs)
    return merge_cheart_meshes(nodal_meshes).next()


def create_centerline_mesh_in_surface[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    in_surf: int,
    a_z: A1[np.floating],
    partition: int | A1[np.floating] | CLPartition[np.floating, np.integer],
    **kwargs: Unpack[CLTopologyKwargs],
) -> Result[MergedMesh[F, I]]:
    if not mesh.bnd:
        msg = "Mesh has no boundary surfaces."
        return Err(ValueError(msg))
    if in_surf not in mesh.bnd.v:
        msg = f"Surface {in_surf} not found in mesh boundary."
        return Err(ValueError(msg))
    surf_nodes = np.unique(mesh.bnd.v[in_surf].v)
    new_az = a_z[surf_nodes]
    match create_mesh_from_surface(mesh, in_surf):
        case Ok(surf_mesh): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    return create_centerline_mesh_in_volume(surf_mesh, new_az, partition, **kwargs).next()


def create_centerline_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    a_z: A1[np.floating],
    partition: int | A1[np.floating] | CLPartition[np.floating, np.integer],
    *,
    in_surf: int | None = None,
) -> Result[MergedMesh[F, I]]:
    """Create a centerline mesh from a given mesh and z-coordinates.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The input mesh.

    a_z : A1[np.floating]
        The z-coordinates of the centerline.

    partition : int | A1[np.floating] | CLPartition[np.floating, np.integer]
        The partition of the centerline.
        -   If int, it is the number of nodes in the centerline.
        -   If A1[np.floating], it is the z-coordinates of the nodes
        -   If CLPartition, it is the partition of the centerline. Returns Self.

    in_surf : int, optional
        The surface index to create the centerline mesh in. If it is not None, the centerline mesh
        will be created in the surface instead over the entire mesh.

    Returns
    -------
    Result[MergedMesh[F, I]]
        The centerline mesh.

    """
    if in_surf is not None:
        return create_centerline_mesh_in_surface(mesh, in_surf, a_z, partition).next()
    return create_centerline_mesh_in_volume(mesh, a_z, partition).next()
