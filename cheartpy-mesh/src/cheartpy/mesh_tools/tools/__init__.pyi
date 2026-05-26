from collections.abc import Mapping, Sequence

import numpy as np
from pytools.arrays import A2
from pytools.result import Result

from cheartpy.mesh import CheartMesh

from ._types import MergedMesh as MergedMesh

def merge_meshes[F: np.floating, I: np.integer](
    meshes: Sequence[CheartMesh[F, I]], vs: Mapping[str, Sequence[A2[F]]]
) -> Result[MergedMesh[F, I]]: ...
