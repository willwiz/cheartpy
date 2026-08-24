import dataclasses as dc
import itertools
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import Gmsh2Vtk, GmshEnum, Vtk2CheartNodeOrder
from cheartpy.mesh import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from pytools.result import Err, Ok, Result, all_ok

import gmsh

if TYPE_CHECKING:
    from cheartpy.elem_interfaces._types import VtkEnum
    from cheartpy.gmsh.types import Entity, Tag
    from pytools.arrays import A1, A2, DType


@dc.dataclass(slots=True)
class GmshSpace[F: np.floating, I: np.integer]:
    dim: int
    tags: A1[I]
    coord: A2[F]


@dc.dataclass(slots=True)
class GmshElements[I: np.integer]:
    type: GmshEnum
    tags: A1[I]
    conn: A2[I]


type ElemSearchMap = Mapping[int, set[int]]


def build_element_searchmap[I: np.integer](k: A1[I], conn: A2[I]) -> ElemSearchMap:
    """Create a mapping to find elements that contain a given node."""
    search_map = defaultdict[int, set[int]](set)
    for elem, nodes in zip(k, conn, strict=True):
        for node in nodes:
            search_map[node].add(int(elem))
    return search_map


def search_element(search_map: ElemSearchMap, node: Iterable[int]) -> Result[int]:
    """Find elements that contain all of the given node."""
    possible_elems = set[int].intersection(*(search_map[n] for n in node))
    if not possible_elems:
        msg = f"No element contains all nodes {node}."
        return Err(ValueError(msg))
    if len(possible_elems) > 1:
        return Err(ValueError(f"Multiple elements contain all nodes {node}: {possible_elems}."))
    return Ok(possible_elems.pop())


def search_entity_by_physical_group(name: str, dim: int) -> list[Entity]:
    entities = gmsh.model.get_entities_for_physical_name(name)
    if not entities:
        msg = f"No entity found for physical group '{name}'."
        raise ValueError(msg)
    return [i for i, d in entities if d == dim]


def get_element[I: np.integer = np.intp](
    el_type: int, tags: Sequence[int], nodes: Sequence[int], *, dtype: DType[I] = np.intp
) -> GmshElements[I]:
    _, _, _, elem_size, _, _ = gmsh.model.mesh.get_element_properties(el_type)
    return GmshElements(
        type=GmshEnum(el_type),
        tags=np.ascontiguousarray(tags, dtype=dtype),
        conn=np.ascontiguousarray(nodes, dtype=dtype).reshape(-1, elem_size),
    )


def merge_elements[I: np.integer](elements: Iterable[GmshElements[I]]) -> GmshElements[I]:
    elements = list(elements)
    if not elements:
        msg = "No elements to merge."
        raise ValueError(msg)
    el_types = {el.type for el in elements}
    if len(el_types) != 1:
        msg = f"Cannot merge elements of different types: {el_types}."
        raise ValueError(msg)
    tags = np.concatenate([el.tags for el in elements])
    conn = np.concatenate([el.conn for el in elements])
    return GmshElements(type=el_types.pop(), tags=tags, conn=conn)


def get_gmsh_entity[I: np.integer = np.intp](
    dim: int, entity_tag: Sequence[Tag], *, dtype: DType[I] = np.intp
) -> GmshElements[I]:
    res = itertools.chain.from_iterable(
        zip(*gmsh.model.mesh.get_elements(dim=dim, tag=i), strict=True) for i in entity_tag
    )
    return merge_elements([get_element(t, tag, nodes, dtype=dtype) for t, tag, nodes in res])


def get_gmsh_entity_by_physical_group[I: np.integer = np.intp](
    name: Sequence[str], dim: int, *, dtype: DType[I] = np.intp
) -> GmshElements[I]:
    entity_tags = [i for s in name for i in search_entity_by_physical_group(s, dim)]
    return get_gmsh_entity(dim, entity_tags, dtype=dtype)


def get_gmsh_space[F: np.floating, I: np.integer](
    dim: int, *, ftype: DType[F] = np.float64, dtype: DType[I] = np.intp
) -> GmshSpace[F, I]:
    node_tags, coordinates, _ = gmsh.model.mesh.get_nodes(dim=dim, tag=-1)
    return GmshSpace(
        dim=dim,
        tags=np.ascontiguousarray(node_tags, dtype=dtype),
        coord=np.ascontiguousarray(coordinates, dtype=ftype).reshape(-1, dim),
    )


@dc.dataclass(slots=True)
class GmshBoundaries[I: np.integer = np.intp]:
    dim: int
    entity: Entity
    type: GmshEnum
    k: A1[I]
    v: A2[I]


def get_gmsh_boundaries[I: np.integer](
    dim: int, entity: Entity, domain: GmshElements[I]
) -> GmshBoundaries[I]:
    bnd = get_gmsh_entity(dim, [entity], dtype=domain.tags.dtype)
    search_map = build_element_searchmap(domain.tags, domain.conn)
    match all_ok([search_element(search_map, nodes) for nodes in bnd.conn]):
        case Ok(top_id): ...  # fmt: skip
        case Err(e):
            raise e
    el_type = GmshEnum(bnd.type)
    return GmshBoundaries(
        dim=dim,
        entity=entity,
        type=el_type,
        k=np.array(top_id, dtype=domain.tags.dtype),
        v=bnd.conn,
    )


def convert_gmsh_space_to_cheart[F: np.floating = np.float64, I: np.integer = np.intp](
    space: GmshSpace[F, I],
) -> CheartMeshSpace[F]:
    nodes = np.zeros((space.tags.max() + 1, space.coord.shape[1]), dtype=space.coord.dtype)
    nodes[space.tags] = space.coord
    return CheartMeshSpace(len(nodes), nodes)


def convert_gmsh_top_to_cheart[I: np.integer = np.intp](
    top: GmshElements[I], *, dtype: DType[I] = np.intp
) -> CheartMeshTopology[I]:
    vtk_type = Gmsh2Vtk[top.type]
    reorder = Vtk2CheartNodeOrder[vtk_type]
    return CheartMeshTopology(
        n=len(top.conn),
        v=np.ascontiguousarray(top.conn[:, reorder] - 1, dtype=dtype),
        TYPE=vtk_type,
    )


def convert_gmsh_bnd_to_cheart_patch[I: np.integer = np.intp](
    bnd: GmshBoundaries[I], tag: int, vtk_type: VtkEnum, *, dtype: DType[I] = np.intp
) -> CheartMeshPatch[I]:
    reorder = Vtk2CheartNodeOrder[vtk_type]
    return CheartMeshPatch(
        tag=tag,
        n=len(bnd.k),
        k=bnd.k - 1,
        v=np.ascontiguousarray(bnd.v[:, reorder] - 1, dtype=dtype),
        TYPE=vtk_type,
    )


def convert_gmsh_bnd_to_cheart[I: np.integer = np.intp](
    boundary: Mapping[int, GmshBoundaries[I]], dtype: DType[I] = np.intp
) -> CheartMeshBoundary[I]:
    vtk_types = {Gmsh2Vtk[b.type] for b in boundary.values()}
    if len(vtk_types) != 1:
        msg = f"All boundaries must have the same type, found: {vtk_types}."
        raise ValueError(msg)
    vtk_type = vtk_types.pop()
    return CheartMeshBoundary(
        n=len(boundary),
        v={
            k: convert_gmsh_bnd_to_cheart_patch(b, tag=k, vtk_type=vtk_type, dtype=dtype)
            for k, b in boundary.items()
        },
        TYPE=vtk_type,
    )


def create_mask_from_domains[I: np.integer](
    domains: Mapping[int, GmshElements[I]], top: GmshElements[I]
) -> A1[I]:
    master_index = np.unique(top.tags, sorted=True)
    if len(master_index) != len(top.tags):
        msg = "Duplicate element tags found in the master topology."
        raise ValueError(msg)
    total_domain_elems = sum(len(d.tags) for d in domains.values())
    merged_domain_tags = np.concatenate([d.tags for d in domains.values()])
    if total_domain_elems != len(merged_domain_tags):
        msg = "Duplicate element tags found across subdomains."
        raise ValueError(msg)
    mask = np.zeros(len(master_index), dtype=top.tags.dtype)
    for domain_id, domain in domains.items():
        mask[np.searchsorted(master_index, domain.tags)] = domain_id
    return mask


def build_cheart_mesh_from_gmsh[F: np.floating, I: np.integer](
    dim: int,
    domains: Mapping[int, Entity],
    boundaries: Mapping[int, tuple[int, Entity]],
    *,
    ftype: DType[F] = np.float64,
    dtype: DType[I] = np.intp,
) -> tuple[CheartMesh[F, I], A1[I] | None]:
    gmsh_space = get_gmsh_space(dim, ftype=ftype, dtype=dtype)
    subdomains = {k: get_gmsh_entity(dim, [v], dtype=dtype) for k, v in domains.items()}
    gmsh_top = merge_elements(subdomains.values())
    gmsh_bnd = {
        k: get_gmsh_boundaries(dim - 1, v, subdomains[d]) for k, (d, v) in boundaries.items()
    }
    space = convert_gmsh_space_to_cheart(gmsh_space)
    top = convert_gmsh_top_to_cheart(gmsh_top, dtype=dtype)
    bnd = convert_gmsh_bnd_to_cheart(gmsh_bnd, dtype=dtype)
    mask = create_mask_from_domains(subdomains, gmsh_top) if len(domains) > 1 else None
    return CheartMesh(space=space, top=top, bnd=bnd), mask
