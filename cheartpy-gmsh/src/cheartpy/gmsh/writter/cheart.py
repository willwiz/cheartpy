import dataclasses as dc
import itertools
from typing import TYPE_CHECKING, overload

import numpy as np
from cheartpy.elem_interfaces import Gmsh2Vtk, GmshEnum, Vtk2CheartNodeOrder
from cheartpy.gmsh.tools import (
    build_element_searchmap,
    find_elem_for_boundary,
)
from cheartpy.mesh import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from pytools.result import Err, Ok, all_ok

import gmsh

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cheartpy.elem_interfaces._types import VtkEnum
    from cheartpy.gmsh.types import Entity, Tag
    from pytools.arrays import A1, A2, DType


@dc.dataclass(slots=True)
class GmshNodes[F: np.floating, I: np.integer]:
    dim: int
    k: A1[I]
    coord: A2[F]


@dc.dataclass(slots=True)
class GmshElements[I: np.integer]:
    type: GmshEnum
    e: A1[I]
    conn: A2[I]


def search_entity_by_physical_name(name: str, dim: int) -> list[Entity]:
    print(f"Searching for physical group '{name}' with dimension {dim}...")
    entities = gmsh.model.get_entities_for_physical_name(name)
    print(f"Found entities: {entities}")
    if not entities:
        msg = f"No entity found for physical group '{name}'."
        raise ValueError(msg)
    return [i for d, i in entities if d == dim]


def get_element[I: np.integer = np.intp](
    el_type: int, tags: Sequence[int], nodes: Sequence[int], *, dtype: DType[I] = np.intp
) -> GmshElements[I]:
    _, _, _, elem_size, _, _ = gmsh.model.mesh.get_element_properties(el_type)
    return GmshElements(
        type=GmshEnum(el_type),
        e=np.ascontiguousarray(tags, dtype=dtype),
        conn=np.ascontiguousarray(nodes, dtype=dtype).reshape(-1, elem_size),
    )


def merge_elements[I: np.integer](elements: Sequence[GmshElements[I]]) -> GmshElements[I]:
    print(f"Merging {len(elements)} GmshElements...")
    elements = list(elements)

    if not elements:
        msg = "No elements to merge."
        raise ValueError(msg)
    el_types = {el.type for el in elements}
    if len(el_types) != 1:
        msg = f"Cannot merge elements of different types: {el_types}."
        raise ValueError(msg)
    tags = np.concatenate([el.e for el in elements])
    conn = np.concatenate([el.conn for el in elements])
    return GmshElements(type=el_types.pop(), e=tags, conn=conn)


def get_gmsh_entity[I: np.integer = np.intp](
    dim: int, entity_tag: Sequence[Tag], *, dtype: DType[I] = np.intp
) -> GmshElements[I]:
    print(f"Getting GMSH entity for dimension {dim} and entity tags {entity_tag}...")
    res = list(
        itertools.chain.from_iterable(
            zip(*gmsh.model.mesh.get_elements(dim=dim, tag=i), strict=True) for i in entity_tag
        )
    )
    return merge_elements([get_element(t, tag, nodes, dtype=dtype) for t, tag, nodes in res])


def get_gmsh_entity_by_physical_group[I: np.integer = np.intp](
    name: Sequence[str], dim: int, *, dtype: DType[I] = np.intp
) -> GmshElements[I]:
    entity_tags = [i for s in name for i in search_entity_by_physical_name(s, dim)]
    print(f"Found entity tags for physical groups {name}: {entity_tags}")
    return get_gmsh_entity(dim, entity_tags, dtype=dtype)


def get_gmsh_space[F: np.floating, I: np.integer](
    dim: int, *, ftype: DType[F] = np.float64, dtype: DType[I] = np.intp
) -> GmshNodes[F, I]:
    node_tags, coordinates, _ = gmsh.model.mesh.get_nodes(dim=dim, tag=-1)
    return GmshNodes(
        dim=dim,
        k=np.ascontiguousarray(node_tags, dtype=dtype),
        coord=np.ascontiguousarray(coordinates, dtype=ftype).reshape(-1, dim),
    )


@dc.dataclass(slots=True)
class GmshBoundaries[I: np.integer = np.intp]:
    dim: int
    entity: Entity
    type: GmshEnum
    e: A1[I]
    conn: A2[I]


def get_gmsh_boundaries[I: np.integer](
    dim: int, entity: Entity, domain: GmshElements[I], k: int
) -> GmshBoundaries[I]:
    print(f"Working on Boundary {k} in dimension {dim}...")
    tags = search_entity_by_physical_name(f"Surface{k}", dim)
    print(f"Surface{k} found Entities: {tags} expect entity {entity}")
    bnd = get_gmsh_entity(dim, [entity], dtype=domain.e.dtype)
    search_map = build_element_searchmap(domain.e, domain.conn)
    match all_ok([find_elem_for_boundary(search_map, nodes, strict=False) for nodes in bnd.conn]):
        case Ok(top_id): ...  # fmt: skip
        case Err(e):
            raise e
    el_type = GmshEnum(bnd.type)
    return GmshBoundaries(
        dim=dim,
        entity=entity,
        type=el_type,
        e=np.array(top_id, dtype=domain.e.dtype),
        conn=bnd.conn,
    )


def convert_gmsh_space_to_cheart[F: np.floating = np.float64, I: np.integer = np.intp](
    space: GmshNodes[F, I],
) -> CheartMeshSpace[F]:
    return CheartMeshSpace(len(space.coord), space.coord)


def convert_gmsh_top_to_cheart[I: np.integer = np.intp](
    top: GmshElements[I], *, dtype: DType[I] = np.intp
) -> CheartMeshTopology[I]:
    vtk_type = Gmsh2Vtk[top.type]
    reorder = Vtk2CheartNodeOrder[vtk_type]
    print(f"Vtk type: {vtk_type}, reorder: {reorder}")
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
        n=len(bnd.e),
        k=bnd.e - 1,
        v=np.ascontiguousarray(bnd.conn[:, reorder] - 1, dtype=dtype),
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
    master_index = np.unique(top.e, sorted=True)
    if len(master_index) != len(top.e):
        msg = "Duplicate element tags found in the master topology."
        raise ValueError(msg)
    total_domain_elems = sum(len(d.e) for d in domains.values())
    merged_domain_tags = np.concatenate([d.e for d in domains.values()])
    if total_domain_elems != len(merged_domain_tags):
        msg = "Duplicate element tags found across subdomains."
        raise ValueError(msg)
    mask = np.zeros(len(master_index), dtype=top.e.dtype)
    for domain_id, domain in domains.items():
        mask[np.searchsorted(master_index, domain.e) - 1] = domain_id
    return mask


@dc.dataclass(slots=True)
class IndexPermutation[I: np.integer]:
    fwd: A1[I]
    inv: A1[I]


def create_index_permutation[I: np.integer](index: A1[I]) -> IndexPermutation[I]:
    perm_inv = index
    perm_fwd = np.full(np.max(perm_inv) + 1, -1, dtype=perm_inv.dtype)
    perm_fwd[perm_inv] = np.arange(1, len(perm_inv) + 1, dtype=perm_inv.dtype)
    return IndexPermutation(fwd=perm_fwd, inv=perm_inv)


def get_index_mapping_permutation[I: np.integer](
    top: GmshElements[I],
) -> IndexPermutation[I]:
    perm_inv = top.e
    perm_fwd = np.full(np.max(perm_inv) + 1, -1, dtype=perm_inv.dtype)
    perm_fwd[perm_inv] = np.arange(1, len(perm_inv) + 1, dtype=perm_inv.dtype)
    return IndexPermutation(perm_fwd, perm_inv)


def reset_node_index[F: np.floating, I: np.integer](
    space: GmshNodes[F, I], top: GmshElements[I], bnd: Mapping[int, GmshBoundaries[I]]
) -> tuple[GmshNodes[F, I], GmshElements[I], Mapping[int, GmshBoundaries[I]]]:
    perm = create_index_permutation(space.k)
    space = GmshNodes(
        dim=space.dim,
        k=np.arange(len(perm.inv), dtype=perm.inv.dtype),
        coord=space.coord,
    )
    top = GmshElements(
        type=top.type,
        e=top.e,
        conn=perm.fwd[top.conn].astype(top.conn.dtype),
    )
    bnd = {
        k: GmshBoundaries(
            dim=v.dim,
            entity=v.entity,
            type=v.type,
            e=v.e,
            conn=perm.fwd[v.conn].astype(v.conn.dtype),
        )
        for k, v in bnd.items()
    }
    return space, top, bnd


@overload
def reset_index_elem[I: np.integer](
    perm: IndexPermutation[I], item: GmshElements[I]
) -> GmshElements[I]: ...
@overload
def reset_index_elem[I: np.integer](
    perm: IndexPermutation[I], item: Mapping[int, GmshElements[I]]
) -> dict[int, GmshElements[I]]: ...
def reset_index_elem[I: np.integer](
    perm: IndexPermutation[I], item: GmshElements[I] | Mapping[int, GmshElements[I]]
):
    if isinstance(item, GmshElements):
        return GmshElements(
            type=item.type,
            e=perm.fwd[item.e].astype(item.e.dtype),
            conn=item.conn,
        )
    return {k: reset_index_elem(perm, v) for k, v in item.items()}


@overload
def reset_index_boundary[I: np.integer](
    perm: IndexPermutation[I], item: GmshBoundaries[I]
) -> GmshBoundaries[I]: ...
@overload
def reset_index_boundary[I: np.integer](
    perm: IndexPermutation[I], item: Mapping[int, GmshBoundaries[I]]
) -> dict[int, GmshBoundaries[I]]: ...
def reset_index_boundary[I: np.integer](
    perm: IndexPermutation[I], item: GmshBoundaries[I] | Mapping[int, GmshBoundaries[I]]
):
    if isinstance(item, GmshBoundaries):
        return GmshBoundaries(
            dim=item.dim,
            entity=item.entity,
            type=item.type,
            e=perm.fwd[item.e].astype(item.e.dtype),
            conn=item.conn,
        )
    return {k: reset_index_boundary(perm, v) for k, v in item.items()}


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
    gmsh_top = merge_elements(list(subdomains.values()))
    # gmsh_top = get_gmsh_entity_by_physical_group(["Volume1"], dim, dtype=dtype)
    gmsh_bnd = {
        k: get_gmsh_boundaries(dim - 1, v, subdomains[d], k) for k, (d, v) in boundaries.items()
    }
    gmsh_space, gmsh_top, gmsh_bnd = reset_node_index(gmsh_space, gmsh_top, gmsh_bnd)
    elem_perm = create_index_permutation(gmsh_top.e)
    gmsh_top = reset_index_elem(elem_perm, gmsh_top)
    gmsh_bnd = reset_index_boundary(elem_perm, gmsh_bnd)
    subdomains = reset_index_elem(elem_perm, subdomains)
    space = convert_gmsh_space_to_cheart(gmsh_space)
    top = convert_gmsh_top_to_cheart(gmsh_top, dtype=dtype)
    bnd = convert_gmsh_bnd_to_cheart(gmsh_bnd, dtype=dtype)
    mask = create_mask_from_domains(subdomains, gmsh_top) if len(domains) > 1 else None
    return CheartMesh(space=space, top=top, bnd=bnd), mask
