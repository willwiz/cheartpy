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
def recompile_cheart_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> CheartMesh[F, I]: ...
def build_index_update_map[I: np.integer](elements: A2[I]) -> IndexUpdateMap: ...
def build_element_searchmap[I: np.integer](elements: Mapping[int, A1[I]]) -> ElemSearchMap: ...
def search_element(search_map: ElemSearchMap, node: Collection[int]) -> Result[set[int]]: ...
def search_element_unique(search_map: ElemSearchMap, node: Collection[int]) -> Result[int]: ...
def search_elements_from_boundary_set[I: np.integer](
    elemnts: A2[I], bnd: A2[I]
) -> Result[Sequence[int]]: ...
