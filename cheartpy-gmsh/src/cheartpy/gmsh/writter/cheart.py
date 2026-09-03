import dataclasses as dc
import itertools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, overload

import numpy as np
from cheartpy.elem_interfaces import (
    CheartEnum,
    GmshEnum,
    convert_element_type,
    get_node_permutation,
)
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
    vtk_type = convert_element_type(top.type, "Cheart")
    perm = get_node_permutation(top.type, "Cheart")
    print(f"Vtk type: {vtk_type}, reorder: {perm}")
    return CheartMeshTopology(
        n=len(top.conn),
        v=np.ascontiguousarray(top.conn[:, perm] - 1, dtype=dtype),
        TYPE=vtk_type,
    )


def convert_gmsh_bnd_to_cheart_patch[I: np.integer = np.intp](
    bnd: GmshBoundaries[I], tag: int, vtk_type: CheartEnum, *, dtype: DType[I] = np.intp
) -> CheartMeshPatch[I]:
    perm = get_node_permutation(bnd.type, "Cheart")
    return CheartMeshPatch(
        tag=tag,
        n=len(bnd.e),
        k=bnd.e - 1,
        v=np.ascontiguousarray(bnd.conn[:, perm] - 1, dtype=dtype),
        TYPE=vtk_type,
    )


def convert_gmsh_bnd_to_cheart[I: np.integer = np.intp](
    boundary: Mapping[int, GmshBoundaries[I]], dtype: DType[I] = np.intp
) -> CheartMeshBoundary[I]:
    vtk_types = {convert_element_type(b.type, "Cheart") for b in boundary.values()}
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


@overload
def search_unique[I: np.integer](a: A1[I], b: A1[I]) -> A1[I]: ...
@overload
def search_unique[I: np.integer](a: A1[I], b: Sequence[A1[I]]) -> Sequence[A1[I]]: ...
@overload
def search_unique[I: np.integer](a: A1[I], b: Mapping[int, A1[I]]) -> Mapping[int, A1[I]]: ...
def search_unique[I: np.integer](
    a: A1[I], b: A1[I] | Sequence[A1[I]] | Mapping[int, A1[I]]
) -> A1[I] | Sequence[A1[I]] | Mapping[int, A1[I]]:
    """Return the position of every index of each array b.

    Assume every element in a and b are unique.

    Parameters
    ----------
    a : A1[I]
        The array to search in.
    b : A1[I] | Sequence[A1[I]] | Mapping[int, A1[I]]
        The arrays to search for.

    Returns
    -------
    A1[I] | Sequence[A1[I]] | Mapping[int, A1[I]]
        The positions of each array b in a.

    """
    sorter = np.argsort(a).astype(a.dtype)
    match b:
        case Mapping():
            return {k: sorter[np.searchsorted(a, v, sorter=sorter)] for k, v in b.items()}
        case Sequence():
            return [sorter[np.searchsorted(a, v, sorter=sorter)] for v in b]
        case np.ndarray():
            return sorter[np.searchsorted(a, b, sorter=sorter)]


def create_mask_from_domains[I: np.integer](
    domains: Mapping[int, GmshElements[I]], top: GmshElements[I]
) -> A1[I]:
    if len(np.unique(top.e)) != len(top.e):
        msg = "Duplicate element tags found in the master topology."
        raise ValueError(msg)
    for k, v in domains.items():
        if len(np.unique(v.e)) != len(v.e):
            msg = f"Duplicate element tags found in subdomain {k}."
            raise ValueError(msg)
        if not np.all(np.isin(top.e, v.e)):
            msg = f"Elements in subdomain {k} is not found in master"

    merged_domain_tags = np.concatenate([d.e for d in domains.values()])
    if len(np.unique(merged_domain_tags)) != len(merged_domain_tags):
        msg = "Duplicate element tags found across subdomains."
        raise ValueError(msg)
    if len(merged_domain_tags) != len(top.e):
        msg = "merged domain tags do not cover all elements in the master topology."
    mask = np.full_like(top.e, -1)
    index = {k: v.e for k, v in domains.items()}
    for domain_id, domain in search_unique(top.e, index).items():
        mask[domain] = domain_id
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
    """Return a CheartMesh and a mask from GMSH entities if given.\

    Parameters
    ----------
    dim : int
        The dimension of the main volume mesh (2 or 3).
    domains: Mapping[int, Entity]
        A mapping of subdomain IDs to GMSH entity tags for the volume mesh. If only one subdomain
        is present, this can be just {1: Entity}.
    boundaries: Mapping[int, tuple[int, Entity]]
        A mapping of domain IDs to a mapping of the desired boundary tag(int) to the GMSH entity.
        mesh.
    ftype : DType[F], optional
        The floating point data type for the mesh coordinates, by default np.float64.
    dtype : DType[I], optional
        The integer data type for the mesh indices, by default np.intp.

    Returns
    -------
    mesh: CheartMesh[F, I]
        A tuple containing the CheartMesh and an optional mask array indicating subdomain
        membership.

    mask: A1[I] | None
        An array of length equal to the number of elements in the mesh, containing integer IDs
        that indicate which subdomain each element belongs to. If only one subdomain is present,
        this will be None.

    """
    gmsh_space = get_gmsh_space(dim, ftype=ftype, dtype=dtype)
    subdomains = {k: get_gmsh_entity(dim, [v], dtype=dtype) for k, v in domains.items()}
    gmsh_top = merge_elements(list(subdomains.values()))
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
