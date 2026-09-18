import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A1, ToIndex

    from cheartpy.mesh import CheartMesh

type ElemSearchMap = Mapping[ToIndex, set[int]]


@dc.dataclass(slots=True)
class IndexPermutation[I: np.integer]:
    old: A1[I]
    new: A1[I]
    fwd: A1[I]


@dc.dataclass(slots=True)
class MergedMesh[F: np.floating, I: np.integer]:
    mesh: CheartMesh[F, I]
    iface: CheartMesh[F, I]
    node_perm: Mapping[int, IndexPermutation[I]]
    elem_perm: Mapping[int, IndexPermutation[I]]
