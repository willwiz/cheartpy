import numpy as np
from pytools.arrays import A1, A2

from cheartpy.mesh import CheartMesh

from ._types import IndexPermutation

def create_index_permutation[I: np.integer](index: A1[I] | A2[I]) -> IndexPermutation[I]: ...
def recompile_cheart_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> CheartMesh[F, I]: ...
