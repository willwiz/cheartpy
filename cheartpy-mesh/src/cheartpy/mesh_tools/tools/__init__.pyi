from collections.abc import Collection, Mapping, Sequence

import numpy as np
from pytools.arrays import A1, A2
from pytools.result import Result

from cheartpy.mesh import CheartMesh

from ._types import ElemSearchMap, IndexUpdateMap
from ._types import MergedMesh as MergedMesh

def merge_meshes[F: np.floating, I: np.integer](
    meshes: Sequence[CheartMesh[F, I]], vs: Mapping[str, Sequence[A2[F]]]
) -> Result[MergedMesh[F, I]]: ...
def build_index_update_map[I: np.integer](elements: Mapping[int, A1[I]]) -> IndexUpdateMap: ...
def build_element_searchmap[I: np.integer](elements: Mapping[int, A1[I]]) -> ElemSearchMap: ...
def search_element(search_map: ElemSearchMap, node: Collection[int]) -> Result[Collection[int]]: ...
