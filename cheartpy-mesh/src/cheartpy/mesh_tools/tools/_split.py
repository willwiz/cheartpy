import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Err, Ok, Result

from cheartpy.mesh import CheartMesh, CheartMeshSpace, CheartMeshTopology
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
