import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np

from cheartpy.mesh import CheartMeshBoundary
from cheartpy.mesh_tools.tools import IndexPermutation, create_index_permutation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1

    from cheartpy.mesh import CheartMesh


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


def create_permutation_for_subdomain[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], mask: A1[I], domains: Sequence[int] | A1[I]
) -> SubdomainPermutation[I]:
    """Return a permutation for subdomain elements and nodes.

    A mask (length of the elements) is needed to provide ids matching the elements to indicate
    which subdomain they belong to. The order of the elements are preserved, the nodes are not.
    """
    index = get_subdomain_index(mask, domains)
    elem_perm = create_index_permutation(index)
    subdomain_elems = mesh.top.v[index]
    subdomain_nodes = np.unique(subdomain_elems)
    node_perm = create_index_permutation(subdomain_nodes)
    return SubdomainPermutation(node_perm, elem_perm)


def get_cheart_mesh_in_region[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], mask: A1[I], domains: Sequence[int] | A1[I]
) -> CheartMesh[F, I]:
    """Return a new mesh with elements in the given region.

    A mask (length of the elements) is needed to provide ids matching the elements to indicate
    which subdomain they belong to. The order of the elements are preserved, the nodes are not.
    """
    # index = get_subdomain_index(mask, domains)
    raise NotImplementedError


def split_subdomains[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], mask: A1[I], domains: Sequence[Sequence[int]]
) -> CheartMesh[F, I]:
    """Return a new mesh with discontinuous subdomains.

    A mask (length of the elements) is needed to provide ids matching the elements to indicate
    which subdomain they belong to. For every list of int IDs in domains variable sets up a
    separate continuous subdomain in the new mesh. The order of the elements are preserved, the
    nodes are not.
    """
    domain_emap = {k: get_subdomain_index(mask, v) for k, v in enumerate(domains)}
    # create an initial node map assuming each subdomain is independent
    domain_nmap = {k: create_index_permutation(mesh.top.v[ix]) for k, ix in domain_emap.items()}
    # find what the initial index would be if the subdomains were concatenated together
    num_nodes = {k: len(v.inv) for k, v in domain_nmap.items()}
    starting_index = np.add.accumulate([0, *list(num_nodes.values())])
    # create a new node map that accounts for the concatenation of the subdomains
    domain_nmap = {
        k: create_index_permutation(mesh.top.v[ix], first=starting_index[k])
        for k, ix in domain_emap.items()
    }
    x = np.concatenate([mesh.space.v[v.inv] for v in domain_nmap.values()], axis=0)
    t = np.full_like(mesh.top.v, -1)
    for k, ix in domain_emap.items():
        t[ix] = domain_nmap[k].fwd[mesh.top.v[ix]]
    b = (
        CheartMeshBoundary(
            mesh.bnd.n,
            {
                k: CheartMeshPatch(v.tag, v.n, v.k, domain_nmap[k].fwd[v.v], v.TYPE)
                for k, v in mesh.bnd.v.items()
            },
            mesh.bnd.TYPE,
        )
        if mesh.bnd
        else None
    )
    raise NotImplementedError
