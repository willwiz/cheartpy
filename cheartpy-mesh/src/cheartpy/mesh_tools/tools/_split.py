import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Err, Ok, Result

from cheartpy.mesh import CheartMesh, CheartMeshPatch, CheartMeshSpace, CheartMeshTopology
from cheartpy.mesh_tools.tools import IndexPermutation, create_index_permutation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1


@dc.dataclass(slots=True)
class SubdomainPermutation[I: np.integer]:
    node: IndexPermutation[I]
    elem: IndexPermutation[I]


def get_subdomain_index[F: np.floating, I: np.integer](
    mask: A1[I], domains: Sequence[int] | A1[I]
) -> A1[I]:
    """Return element indices for subdomain elements."""
    index, _ = np.where(np.isin(mask, domains))
    return index.astype(mask.dtype)


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
    x = np.concatenate([mesh.space.v[v.idx] for v in domain_nmap.values()], axis=0)
    top = CheartMeshTopology(n=len(mesh.top.v), v=np.full_like(mesh.top.v, -1), TYPE=mesh.top.TYPE)
    for k, ix in domain_emap.items():
        top.v[ix] = domain_nmap[k].fwd[mesh.top.v[ix]]
    if np.any(top.v == -1):
        return Err(ValueError("Some elements were not assigned to any subdomain."))
    return Ok(CheartMesh(space=CheartMeshSpace(n=len(x), v=x), top=top, bnd=None))


def create_mesh_from_surface[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], surf_id: int
) -> Result[CheartMesh[F, I]]:
    """Create a new cheart mesh from a surface mesh.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The input mesh to create a surface mesh from.
    surf_id : int
        The ID of the surface in the boundary of the mesh.

    Returns
    -------
    CheartMesh[F, I]
        A new mesh containing only the surface defined by the boundary.

    """
    if mesh.bnd is None:
        msg = "Mesh has no boundary, cannot create surface mesh"
        return Err(ValueError(msg))
    if (surface := mesh.bnd.v.get(surf_id)) is None:
        return Err(ValueError(f"Boundary {surf_id} not found in mesh"))
    perm = create_index_permutation(surface.v)
    space = CheartMeshSpace(n=len(perm.idx), v=mesh.space.v[perm.idx])
    top = CheartMeshTopology(n=surface.n, v=perm.fwd[surface.v], TYPE=surface.TYPE)
    return Ok(CheartMesh(space=space, top=top, bnd=None))


def filter_boundary_by_elements[F: np.floating, I: np.integer](
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
    subset = np.isin(patch.v, elements)
    if not np.any(subset):
        return None
    perm = create_index_permutation(elements)
    return CheartMeshPatch(
        tag=patch.tag, n=np.sum(subset), k=perm.fwd[patch.k], v=patch.v[subset], TYPE=patch.TYPE
    )


def create_mesh_from_region[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], mask: A1[I], region_id: int
) -> Result[CheartMesh[F, I]]:
    """Create a new cheart mesh from a region in the input mesh.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The input mesh to create a region mesh from.
    mask : A1[I]
        An array of length equal to the number of elements in the mesh, containing integer IDs
        that indicate which region each element belongs to.
    region_id : int
        The ID of the region in the input mesh.

    Returns
    -------
    Result[CheartMesh[F, I]]
        A new mesh containing only the region defined by the region ID.

    """
    e_index = get_subdomain_index(mask, [region_id])
    elements = mesh.top.v[e_index]
    perm = create_index_permutation(elements)
    space = CheartMeshSpace(n=len(perm.idx), v=mesh.space.v[perm.idx])
    top = CheartMeshTopology(n=perm.fwd.shape[0], v=perm.fwd[elements], TYPE=mesh.top.TYPE)
    if mesh.bnd is None:
        return Ok(CheartMesh(space=space, top=top, bnd=None))
    bnd_patches = {k: filter_boundary_by_elements(v, e_index) for k, v in mesh.bnd.v.items()}
    bnd_patches = {
        k: CheartMeshPatch(tag=v.tag, n=v.n, k=v.k, v=perm.fwd[v.v], TYPE=v.TYPE)
        for k, v in bnd_patches.items()
        if v is not None
    }
    return Ok(CheartMesh(space=space, top=top, bnd=None))
