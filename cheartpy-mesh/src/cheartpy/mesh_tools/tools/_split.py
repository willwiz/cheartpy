from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1

    from cheartpy.mesh import CheartMesh


def get_cheart_mesh_in_region[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], mask: A1[I], domains: Sequence[int] | A1[I]
) -> CheartMesh[F, I]:
    """Return a new mesh with elements in the given region.

    A mask (length of the elements) is needed to provide ids matching the elements to indicate
    which subdomain they belong to. The order of the elements are preserved, the nodes are not.
    """
    index = get_subdomain_index(mask, domains)
    raise NotImplementedError


def get_subdomain_index[F: np.floating, I: np.integer](
    mask: A1[I], domains: Sequence[int] | A1[I]
) -> A1[I]:
    """Return element indices for subdomain elements."""
    index, _ = np.where(np.isin(mask, domains))
    return index.astype(mask.dtype)


def split_subdomains[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], mask: A1[I], domains: Sequence[Sequence[int]]
) -> CheartMesh[F, I]:
    """Return a new mesh with discontinuous subdomains.

    A mask (length of the elements) is needed to provide ids matching the elements to indicate
    which subdomain they belong to. For every list of int IDs in domains variable sets up a
    separate continuous subdomain in the new mesh. The order of the elements are preserved, the
    nodes are not.
    """
    domain_masks = [get_subdomain_index(mask, domain) for domain in domains]
    raise NotImplementedError
