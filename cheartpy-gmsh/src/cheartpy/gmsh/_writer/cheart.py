import dataclasses as dc
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
    GmshElements,
    build_element_searchmap,
    find_elem_for_boundary,
    get_gmsh_entity,
    merge_elements,
)
from cheartpy.mesh import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from cheartpy.mesh_tools import IndexPermutation, create_index_permutation
from pytools.result import Err, Ok, all_ok

import gmsh

if TYPE_CHECKING:
    from cheartpy.gmsh.types import Entity
    from pytools.arrays import A1, A2, DType


@dc.dataclass(slots=True)
class GmshNodes[F: np.floating, I: np.integer]:
    dim: int
    k: A1[I]
    coord: A2[F]


def search_entity_by_physical_name(name: str, dim: int) -> list[Entity]:
    entities = gmsh.model.get_entities_for_physical_name(name)
    if not entities:
        msg = f"No entity found for physical group '{name}'."
        raise ValueError(msg)
    return [i for d, i in entities if d == dim]


def get_gmsh_entity_by_physical_group[I: np.integer = np.intp](
    name: Sequence[str], dim: int, *, dtype: DType[I] = np.intp
) -> GmshElements[I]:
    entity_tags = [i for s in name for i in search_entity_by_physical_name(s, dim)]
    return get_gmsh_entity(dim, entity_tags, dtype=dtype)


def get_gmsh_space[F: np.floating, I: np.integer](
    dim: int, *, ftype: DType[F] = np.float64, dtype: DType[I] = np.intp
) -> GmshNodes[F, I]:
    node_tags, coordinates, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
    return GmshNodes(
        dim=dim,
        k=np.ascontiguousarray(node_tags - 1, dtype=dtype),
        coord=np.ascontiguousarray(coordinates, dtype=ftype).reshape(-1, dim),
    )


@dc.dataclass(slots=True)
class GmshBoundaries[I: np.integer = np.intp]:
    dim: int
    entity: Entity
    kind: GmshEnum
    e: A1[I]
    conn: A2[I]


def get_gmsh_boundaries[I: np.integer](
    dim: int, entity: Entity, domain: GmshElements[I]
) -> GmshBoundaries[I]:
    # tags = search_entity_by_physical_name(f"Surface{entity}", dim)
    bnd = get_gmsh_entity(dim, [entity], dtype=domain.e.dtype)
    search_map = build_element_searchmap(domain.e, domain.conn)
    match all_ok([find_elem_for_boundary(search_map, nodes, strict=False) for nodes in bnd.conn]):
        case Ok(top_id): ...  # fmt: skip
        case Err(e):
            raise e
    el_type = GmshEnum(bnd.kind)
    return GmshBoundaries(
        dim=dim,
        entity=entity,
        kind=el_type,
        e=np.array(top_id, dtype=domain.e.dtype),
        conn=bnd.conn,
    )


def convert_gmsh_space_to_cheart[F: np.floating = np.float64, I: np.integer = np.intp](
    space: GmshNodes[F, I],
) -> CheartMeshSpace[F]:
    return CheartMeshSpace(space.coord)


def convert_gmsh_top_to_cheart[I: np.integer = np.intp](
    top: GmshElements[I], *, dtype: DType[I] = np.intp
) -> CheartMeshTopology[I]:
    vtk_type = convert_element_type(top.kind, "Cheart")
    perm = get_node_permutation(top.kind, "Cheart")
    return CheartMeshTopology(v=np.ascontiguousarray(top.conn[:, perm], dtype=dtype), kind=vtk_type)


def convert_gmsh_bnd_to_cheart_patch[I: np.integer = np.intp](
    bnd: GmshBoundaries[I], tag: int, vtk_type: CheartEnum, *, dtype: DType[I] = np.intp
) -> CheartMeshPatch[I]:
    perm = get_node_permutation(bnd.kind, "Cheart")
    return CheartMeshPatch(
        tag=tag, k=bnd.e, v=np.ascontiguousarray(bnd.conn[:, perm], dtype=dtype), kind=vtk_type
    )


def convert_gmsh_bnd_to_cheart[I: np.integer = np.intp](
    boundary: Mapping[int, GmshBoundaries[I]], dtype: DType[I] = np.intp
) -> CheartMeshBoundary[I]:
    vtk_types = {convert_element_type(b.kind, "Cheart") for b in boundary.values()}
    if len(vtk_types) != 1:
        msg = f"All boundaries must have the same type, found: {vtk_types}."
        raise ValueError(msg)
    vtk_type = vtk_types.pop()
    return CheartMeshBoundary(
        v={
            k: convert_gmsh_bnd_to_cheart_patch(b, tag=k, vtk_type=vtk_type, dtype=dtype)
            for k, b in boundary.items()
        }
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
    sorter = np.asarray(np.argsort(a), a.dtype)
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


def reset_node_index[F: np.floating, I: np.integer](
    space: GmshNodes[F, I], top: GmshElements[I], bnd: Mapping[int, GmshBoundaries[I]]
) -> tuple[GmshNodes[F, I], GmshElements[I], Mapping[int, GmshBoundaries[I]]]:
    perm = create_index_permutation(top.conn)
    space = GmshNodes(
        dim=space.dim,
        k=np.arange(len(perm.new), dtype=perm.new.dtype),
        coord=space.coord[perm.old],
    )
    top = GmshElements(
        kind=top.kind,
        e=top.e,
        conn=perm.fwd[top.conn],
    )
    bnd = {
        k: GmshBoundaries(
            dim=v.dim,
            entity=v.entity,
            kind=v.kind,
            e=v.e,
            conn=perm.fwd[v.conn],
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
            kind=item.kind,
            e=perm.fwd[item.e],
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
            kind=item.kind,
            e=perm.fwd[item.e],
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
    """Return a CheartMesh and a mask from GMSH entities if given.

    Type Parameters
    ---------------
    F : np.floating = np.float64
    I : np.integer = np.intp

    Parameters
    ----------
    dim : int
        The dimension of the main volume mesh (2 or 3).
    domains: Mapping[int, Entity]
        **Mapping[Vol_Tag: int, Entity: int]**

        A mapping of subdomain IDs to GMSH entity tags for the volume mesh. If only one subdomain
        is present, this can be just {1: Entity}.
    boundaries: Mapping[int, tuple[int, Entity]]
        **Mapping[Bnd_Tag: int, (Vol_Tag: int, Entity: int)]**

        A mapping of desired surface IDs to a mapping of the domain to the GMSH entity.
        mesh.
    ftype : DType[F], default=np.float64
        The floating point data type for the mesh coordinates, by default np.float64.
    dtype : DType[I], default=np.intp
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
        b: get_gmsh_boundaries(dim - 1, v, subdomains[d]) for b, (d, v) in boundaries.items()
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
