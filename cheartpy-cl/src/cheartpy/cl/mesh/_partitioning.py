import dataclasses as dc
from typing import TYPE_CHECKING, TypedDict, Unpack

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

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A1, A2, DType
    from pytools.result import Result


@dc.dataclass(slots=True, frozen=True)
class CLPartition[F: np.floating = np.float64]:
    n: int
    nodes: A1[F]
    domain: A2[F]

    @property
    def dtype(self) -> np.dtype[F]:
        return self.nodes.dtype

    def astype[T: np.floating](self, dtype: np.dtype[T]) -> CLPartition[T]:
        return CLPartition(
            n=self.n,
            nodes=np.asarray(self.nodes, dtype=dtype),
            domain=np.asarray(self.domain, dtype=dtype),
        )


def private_create_centerline_partition[F: np.floating](
    param: int | A1[F] | CLPartition[F], /, dtype: DType[F] = np.float64
) -> CLPartition[F]:
    match param:
        case CLPartition():
            return param
        case int() as n:
            nodes = np.linspace(0, 1, n, dtype=dtype)
        case np.ndarray() as nodes: ...  # fmt: skip
    elem_size = np.pad(np.diff(nodes), (1, 1), mode="edge")
    domain = np.stack([nodes - elem_size[:-1] / 2, nodes + elem_size[1:] / 2], axis=1, dtype=dtype)
    return CLPartition(n=len(nodes), nodes=nodes, domain=domain)


def create_centerline_partition[F: np.floating](
    param: int | A1[F], /, dtype: DType[F] = np.float64
) -> CLPartition[F]:
    return private_create_centerline_partition(param, dtype=dtype)


class CLTopologyKwargs(TypedDict, total=False):
    search_map: ElemSearchMap


def create_mesh_for_cl_node[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], a_z: A1[F], domain: A1[F], **kwargs: Unpack[CLTopologyKwargs]
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
    a_z: A1[F],
    partition: int | A1[F] | CLPartition[F],
    **kwargs: Unpack[CLTopologyKwargs],
) -> Mapping[int, CheartMesh[F, I]]:
    search_map = kwargs.get("search_map") or build_element_searchmap(mesh.top.v).unwrap()
    partition = private_create_centerline_partition(partition, dtype=a_z.dtype)
    return {
        i: create_mesh_for_cl_node(mesh, a_z, domain, search_map=search_map)
        for i, domain in enumerate(partition.domain)
    }


def create_centerline_mesh_in_volume[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    a_z: A1[F],
    partition: int | A1[F] | CLPartition[F],
    **kwargs: Unpack[CLTopologyKwargs],
) -> Result[MergedMesh[F, I]]:
    nodal_meshes = create_centerline_nodal_meshes(mesh, a_z, partition, **kwargs)
    return merge_cheart_meshes(nodal_meshes).next()


def create_centerline_mesh_in_surface[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    in_surf: int,
    a_z: A1[F],
    partition: int | A1[F] | CLPartition[F],
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
    a_z: A1[F],
    partition: int | A1[F] | CLPartition[F],
    *,
    in_surf: int | None = None,
) -> Result[MergedMesh[F, I]]:
    if in_surf is not None:
        return create_centerline_mesh_in_surface(mesh, in_surf, a_z, partition).next()
    return create_centerline_mesh_in_volume(mesh, a_z, partition).next()
