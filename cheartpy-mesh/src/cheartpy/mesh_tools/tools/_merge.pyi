from collections.abc import Mapping, Sequence

import numpy as np
from pytools.arrays import A2
from pytools.result import Result

from cheartpy.mesh import CheartMesh

from ._types import MergedMesh

def merge_cheart_meshes[F: np.floating, I: np.integer](
    meshes: Mapping[int, CheartMesh[F, I]] | Sequence[CheartMesh[F, I]],
) -> Result[MergedMesh[F, I]]: ...
def merge_point_variables[F: np.floating, I: np.integer, V: np.floating](
    mesh: MergedMesh[F, I],
    variables: Mapping[int, Mapping[str, A2[V]]] | Mapping[str, Mapping[int, A2[V]]],
) -> Result[Mapping[str, A2[V]]]: ...
def merge_cell_variables[F: np.floating, I: np.integer, V: np.floating](
    mesh: MergedMesh[F, I],
    variables: Mapping[int, Mapping[str, A2[V]]] | Mapping[str, Mapping[int, A2[V]]],
) -> Result[Mapping[str, A2[V]]]: ...
