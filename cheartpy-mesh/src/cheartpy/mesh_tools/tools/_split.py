import dataclasses as dc
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from pytools.result import Err, Ok, Result

from cheartpy.mesh import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)

from ._search import find_elements
from ._validation import create_index_permutation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1, A2

    from ._types import IndexPermutation


@dc.dataclass(slots=True)
class SubdomainPermutation[I: np.integer]:
    node: IndexPermutation[I]
    elem: IndexPermutation[I]


def get_subdomain_index[F: np.floating, I: np.integer](
    mask: A1[I], domains: Sequence[int] | A1[I]
) -> A1[I]:
    """Return element indices for subdomain elements."""
    index, *_ = np.where(np.isin(mask, domains))
    return np.asarray(index, mask.dtype)


def split_subdomains[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], mask: A1[I], domains: Sequence[Sequence[int]]
) -> Result[CheartMesh[F, I]]:
    """Return a new mesh with discontinuous subdomains.

    A mask (length of the elements) is needed to provide ids matching the elements to indicate
    which subdomain they belong to. For every list of int IDs in domains variable sets up a
    separate continuous subdomain in the new mesh. The order of the elements are preserved, the
    nodes are not.

    NOTE: currently does not support boundary patches. There's no good way to handle boundary
    patches that are split between subdomains.


    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The input mesh to split into subdomains.
    mask : A1[I]
        An array of length equal to the number of elements in the mesh, containing integer IDs
        that indicate which subdomain each element belongs to.
    domains : Sequence[Sequence[int]]
        A sequence of sequences, where each inner sequence contains the integer IDs that define
        a subdomain. Each inner sequence corresponds to a separate subdomain in the new mesh.

    Returns
    -------
    CheartMesh[F, I]
        A new mesh with discontinuous subdomains, where the elements are ordered according to the
        input domains, and the nodes are renumbered to be continuous across subdomains.

    """
    domain_emap = {k: get_subdomain_index(mask, v) for k, v in enumerate(domains)}
    # find what the initial index would be if the subdomains were concatenated together
    num_nodes = {k: len(np.unique(mesh.top.v[ix])) for k, ix in domain_emap.items()}
    starting_index = np.add.accumulate([0, *list(num_nodes.values())])
    # create a new node map that accounts for the concatenation of the subdomains
    domain_nmap = {
        k: create_index_permutation(mesh.top.v[ix], first=starting_index[k])
        for k, ix in domain_emap.items()
    }
    x = np.concatenate([mesh.space.v[v.old] for v in domain_nmap.values()], axis=0)
    top = CheartMeshTopology(v=np.full_like(mesh.top.v, -1), type=mesh.top.type)
    for k, ix in domain_emap.items():
        top.v[ix] = domain_nmap[k].fwd[mesh.top.v[ix]]
    if np.any(top.v == -1):
        return Err(ValueError("Some elements were not assigned to any subdomain."))
    return Ok(CheartMesh(space=CheartMeshSpace(x), top=top, bnd={}))


@overload
def create_mesh_from_surface[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], surf_id: int
) -> Result[CheartMesh[F, I]]: ...
@overload
def create_mesh_from_surface[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], surf_id: int, *, return_perm: Literal[True]
) -> Result[tuple[CheartMesh[F, I], IndexPermutation[I]]]: ...
def create_mesh_from_surface[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], surf_id: int, *, return_perm: bool = False
):
    """Create a new cheart mesh from a surface mesh.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The input mesh to create a surface mesh from.
    surf_id : int
        The ID of the surface in the boundary of the mesh.
    return_perm : bool, default=False
        If True, return the index permutation used to create the new mesh. If False, return just
        the mesh.

    Returns
    -------
    CheartMesh[F, I]
        A new mesh containing only the surface defined by the boundary.

    """
    if not mesh.bnd:
        msg = "Mesh has no boundary, cannot create surface mesh"
        return Err(ValueError(msg))
    if (surface := mesh.bnd.v.get(surf_id)) is None:
        return Err(ValueError(f"Boundary {surf_id} not found in mesh"))
    perm = create_index_permutation(surface.v)
    space = CheartMeshSpace(mesh.space.v[perm.old])
    top = CheartMeshTopology(v=perm.fwd[surface.v], type=surface.type)
    if return_perm:
        return Ok((CheartMesh(space=space, top=top, bnd={}), perm))
    return Ok(CheartMesh(space=space, top=top, bnd={}))


def _filter_boundary_by_elements[F: np.floating, I: np.integer](
    patch: CheartMeshPatch[I], elements: A1[I]
) -> CheartMeshPatch[I] | None:
    """Filter a boundary patch to only include elements that are in the provided list of elements.

    Parameters
    ----------
    patch : CheartMeshPatch[I]
        The boundary patch to filter.
    elements : A1[I]
        The list of elements to include in the filtered patch.

    Returns
    -------
    CheartMeshPatch[I] | None
        A new boundary patch containing only the elements in the provided list, or None if no
        elements in the patch are in the provided list.

    """
    subset = np.isin(patch.k, elements)
    if not np.any(subset):
        return None
    perm = create_index_permutation(elements)
    return CheartMeshPatch(
        tag=patch.tag,
        k=perm.fwd[patch.k[subset]],
        v=patch.v[subset],
        type=patch.type,
    )


def _filter_boundary_by_nodes[F: np.floating, I: np.integer](
    patch: CheartMeshPatch[I], vol_connectivity: A2[I]
) -> CheartMeshPatch[I] | None:
    """Filter a boundary patch to only include nodes that are in the provided volume connectivity.

    Parameters
    ----------
    patch : CheartMeshPatch[I]
        The boundary patch to filter.
    vol_connectivity : A2[I]
        The volume connectivity to use for filtering the boundary patch.

    Returns
    -------
    CheartMeshPatch[I] | None
        A new boundary patch containing only the nodes in the provided volume connectivity, or None
        if no nodes in the patch are in the provided volume connectivity.

    """
    elem_index = find_elements(vol_connectivity, patch.v, unique=True)
    patches = {k.unwrap(): v for k, v in zip(elem_index, patch.v, strict=True) if isinstance(k, Ok)}
    if not patches:
        return None
    return CheartMeshPatch(
        tag=patch.tag,
        k=np.array(list(patches.keys()), dtype=patch.k.dtype),
        v=np.array(list(patches.values()), dtype=patch.v.dtype),
        type=patch.type,
    )


@overload
def create_mesh_from_region[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], mask: A1[I], region_id: Sequence[int], *, recalc: bool = False
) -> Result[CheartMesh[F, I]]: ...
@overload
def create_mesh_from_region[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    mask: A1[I],
    region_id: Sequence[int],
    *,
    recalc: bool = False,
    return_perm: Literal[True],
) -> Result[tuple[CheartMesh[F, I], IndexPermutation[I]]]: ...
def create_mesh_from_region[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    mask: A1[I],
    region_id: Sequence[int],
    *,
    recalc: bool = False,
    return_perm: bool = False,
) -> Result[CheartMesh[F, I]] | Result[tuple[CheartMesh[F, I], IndexPermutation[I]]]:
    """Create a new cheart mesh from a region in the input mesh.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The input mesh to create a region mesh from.
    mask : A1[I]
        An array of length equal to the number of elements in the mesh, containing integer IDs
        that indicate which region each element belongs to.
    region_id : Sequence[int]
        The IDs of the region in the input mesh.
    recalc : bool, default=False
        If True, the boundary patches will be recalculated based on the new mesh. If False, the
        boundary patches will be filtered based on the elements in the new mesh.
    return_perm : bool, default=False
        If True, return the index permutation used to create the new mesh. If False, return the
        mesh only.

    Returns
    -------
    Result[CheartMesh[F, I]]
        A new mesh containing only the region defined by the region ID.

    """
    e_index = get_subdomain_index(mask, region_id)
    elements = mesh.top.v[e_index]
    perm = create_index_permutation(elements)
    space = CheartMeshSpace(v=mesh.space.v[perm.old])
    top = CheartMeshTopology(v=perm.fwd[elements], type=mesh.top.type)
    if not mesh.bnd:
        return Ok(CheartMesh(space=space, top=top, bnd={}))
    if recalc:
        bnd_patches = {k: _filter_boundary_by_nodes(v, elements) for k, v in mesh.bnd.v.items()}
    else:
        bnd_patches = {k: _filter_boundary_by_elements(v, e_index) for k, v in mesh.bnd.v.items()}
    bnd_patches = {
        k: CheartMeshPatch(tag=v.tag, k=v.k, v=perm.fwd[v.v], type=v.type)
        for k, v in bnd_patches.items()
        if v is not None
    }
    new_mesh = CheartMesh(space=space, top=top, bnd=CheartMeshBoundary(v=bnd_patches))
    if return_perm:
        return Ok((new_mesh, perm))
    return Ok(new_mesh)
