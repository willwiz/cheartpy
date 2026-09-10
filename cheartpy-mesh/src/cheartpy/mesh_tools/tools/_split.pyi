from collections.abc import Sequence

import numpy as np
from pytools.arrays import A1
from pytools.result import Result

from cheartpy.mesh import CheartMesh

from ._split import SubdomainPermutation as SubdomainPermutation

def split_subdomains[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], mask: A1[I], domains: Sequence[Sequence[int]]
) -> Result[CheartMesh[F, I]]: ...
def create_mesh_from_surface[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], surf_id: int
) -> Result[CheartMesh[F, I]]: ...
def create_mesh_from_region[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], mask: A1[I], region_id: Sequence[int], *, recalc: bool = False
) -> Result[CheartMesh[F, I]]: ...
