from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1

    from cheartpy.mesh import CheartMesh


def split_mesh_domains[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], mask: A1[I], domains: Sequence[Sequence[int]]
) -> CheartMesh[F, I]:
    """Return a new mesh with discontinuous subdomains.

    A mask (length of the elements) is needed to provide ids matching the elements to indicate
    which subdomain they belong to. For every list of int IDs in domains variable sets up a
    separate continuous subdomain in the new mesh. The order of the elements are preserved, the
    nodes are not.
    """
    raise NotImplementedError
