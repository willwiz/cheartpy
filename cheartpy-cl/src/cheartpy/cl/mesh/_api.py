import dataclasses as dc
from collections import defaultdict
from typing import TYPE_CHECKING, Unpack

import numpy as np
from cheartpy.io import chwrite_d_utf
from cheartpy.mesh import CheartMesh, CheartMeshSpace, CheartMeshTopology
from cheartpy.mesh_tools.surface_core import create_mesh_from_surface
from pytools.result import Err, Ok, Result
from typing_extensions import TypedDict

from ._types import APIKwargs, CLDef, CLMesh, CLPartition

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from pytools.arrays import A1


def create_cl_partition[F: np.floating](defn: CLDef[F]) -> CLPartition[F]:
    ftype = defn["a_z"].dtype
    match defn:
        case {"nodes": nodes}: ...  # fmt: skip
        case {"n": nn}:
            nodes = np.linspace(0.0, 1.0, nn, dtype=ftype)
        case _:
            msg = "Unreachable"
            raise RuntimeError(msg)
    extended_nodes = np.array(
        [nodes[0] - nodes[1], *nodes, nodes[-1] + (nodes[-1] - nodes[-2])], dtype=ftype
    )
    domain = np.array(
        [
            [i, j, k]
            for i, j, k in zip(
                extended_nodes[:], extended_nodes[1:], extended_nodes[2:], strict=False
            )
        ],
        dtype=ftype,
    )
    return CLPartition(
        prefix=defn["prefix"]["prefix"],
        in_surf=0,
        node=nodes,
        domain=domain,
    )


@dc.dataclass(slots=True)
class Node2ElemMap[I: np.integer]:
    key: A1[I]
    n2e: Mapping[int, Collection[int]]


def create_node2elem_map[F: np.floating, I: np.integer](mesh: CheartMesh[F, I]) -> Node2ElemMap[I]:
    n2p_map = defaultdict[int, set[int]](set)
    for k, vs in enumerate(mesh.top.v):
        for v in vs:
            n2p_map[int(v)].add(k)
    space_key = np.fromiter(sorted(n2p_map.keys()), dtype=mesh.top.v.dtype)
    return Node2ElemMap(space_key, n2p_map)


class CLTopologyKwargs[I: np.integer](TypedDict, total=False):
    node_map: Node2ElemMap[I]


def create_node_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    a_z: A1[F],
    partition: A1[F],
    **kwargs: Unpack[CLTopologyKwargs[I]],
) -> CheartMesh[F, I]:
    node_map = kwargs.get("node_map") or create_node2elem_map(mesh)
    nodes = node_map.key[(a_z >= partition[0]) & (a_z <= partition[-1])]
    elems = mesh.top.v[
        np.unique(
            np.fromiter({e for n in nodes for e in node_map.n2e[int(n)]}, dtype=mesh.top.v.dtype)
        )
    ]
    expanded_nodelist = np.unique(elems)
    nodex = {int(n): i for i, n in enumerate(expanded_nodelist)}
    elems = np.array([[nodex[int(v)] for v in vs] for vs in elems], dtype=mesh.top.v.dtype)
    return CheartMesh(
        space=CheartMeshSpace(len(expanded_nodelist), mesh.space.v[expanded_nodelist]),
        top=CheartMeshTopology(len(elems), elems, TYPE=mesh.top.TYPE),
        bnd=None,
    )


def assemble_cl_node_meshes[F: np.floating, I: np.integer](
    nodal_meshes: Mapping[int, CheartMesh[F, I]],
) -> Result[CheartMesh[F, I]]:
    ftype = {m.space.v.dtype for m in nodal_meshes.values()}.pop()
    itype = {m.top.v.dtype for m in nodal_meshes.values()}.pop()
    node_offset = np.add.accumulate(
        np.insert([len(x.space.v) for x in nodal_meshes.values()], 0, 0)
    )
    cl_1_x = np.vstack([x.space.v for x in nodal_meshes.values()], dtype=ftype)
    cl_1_t = np.vstack(
        [x.top.v + i for x, i in zip(nodal_meshes.values(), node_offset, strict=False)],
        dtype=itype,
    )
    elem_types = {m.top.TYPE for m in nodal_meshes.values()}
    if len(elem_types) != 1:
        msg = f"Nodal meshes do not have the same type. cannot be combined: \n{elem_types}"
        return Err(ValueError(msg))
    return Ok(
        CheartMesh(
            CheartMeshSpace(len(cl_1_x), cl_1_x),
            CheartMeshTopology(len(cl_1_t), cl_1_t, elem_types.pop()),
            None,
        )
    )


def assemble_interface_mesh[F: np.floating, I: np.integer](
    cl_top: CLPartition[F],
    nodal_meshes: Mapping[int, CheartMesh[F, I]],
) -> Result[CheartMesh[F, I]]:
    dtypes = {x.top.v.dtype for x in nodal_meshes.values()}
    if len(dtypes) != 1:
        msg = f"Nodal meshes do not have the same dtype. cannot be combined: \n{dtypes}"
        return Err(ValueError(msg))
    ftypes = {x.space.v.dtype for x in nodal_meshes.values()}
    if len(ftypes) != 1:
        msg = f"Nodal meshes do not have the same float dtype. cannot be combined: \n{ftypes}"
        return Err(ValueError(msg))
    cl_i_x = np.ascontiguousarray(
        [[c] for _, c, _ in cl_top.domain],
        dtype=ftypes.pop(),
    )
    cl_i_t = np.vstack(
        [np.full((x.top.n, 1), i) for i, x in enumerate(nodal_meshes.values())],
        dtype=dtypes.pop(),
    )
    elem_types = {m.top.TYPE for m in nodal_meshes.values()}
    if len(elem_types) != 1:
        msg = f"Nodal meshes do not have the same type. cannot be combined: \n{elem_types}"
        return Err(ValueError(msg))
    return Ok(
        CheartMesh(
            CheartMeshSpace(len(cl_i_x), cl_i_x),
            CheartMeshTopology(len(cl_i_t), cl_i_t, elem_types.pop()),
            None,
        )
    )


def create_centerline_topology[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], defn: CLDef[F], **kwargs: Unpack[APIKwargs[F]]
) -> Result[CLMesh[F, I]]:
    partition = kwargs.get("partition") or create_cl_partition(defn)
    node_map = create_node2elem_map(mesh)
    node_mesh = {
        k: create_node_mesh(mesh, defn["a_z"], domain, node_map=node_map)
        for k, domain in enumerate(partition.domain)
    }
    match assemble_cl_node_meshes(node_mesh):
        case Ok(volume_top): ...  # fmt: skip
        case Err(e):
            return Err(e)
    match assemble_interface_mesh(partition, node_mesh):
        case Ok(interface_top): ...  # fmt: skip
        case Err(e):
            return Err(e)
    domain = partition.domain
    elem = np.eye(len(domain), dtype=domain.dtype)
    return Ok(CLMesh(volume_top, interface_top, domain, elem))


def create_centerline_topology_in_surf[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], in_surf: int, defn: CLDef[F], **kwargs: Unpack[APIKwargs[F]]
) -> Result[CLMesh[F, I]]:
    if mesh.bnd is None:
        msg = "Mesh does not have boundary."
        return Err(ValueError(msg))
    surf_nodes = np.unique(mesh.bnd.v[in_surf].v)
    a_z = defn["a_z"][surf_nodes]
    match create_mesh_from_surface(mesh, in_surf):
        case Ok(surf_mesh): ...  # fmt: skip
        case Err(e):
            return Err(e)
    defn["a_z"] = a_z
    return create_centerline_topology(surf_mesh, defn, **kwargs).next()


def export_cl_mesh[F: np.floating, I: np.integer](mesh: CLMesh[F, I], defn: CLDef[F]) -> None:
    root = defn["home"]
    p = defn["prefix"]
    mesh.body.save(root / f"{p['prefix']}{p.get('body') or 'Body'}")
    mesh.iface.save(root / f"{p['prefix']}{p.get('iface') or 'IFace'}")
    chwrite_d_utf(root / f"{p['prefix']}{p.get('domain') or 'Domain'}-0.D", mesh.domain)
    chwrite_d_utf(root / f"{p['prefix']}{p.get('elem') or 'Elem'}-0.D", mesh.elem)
