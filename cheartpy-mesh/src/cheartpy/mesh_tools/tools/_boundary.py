import enum
from typing import TYPE_CHECKING

import numpy as np
from pytools.logging import get_logger

if TYPE_CHECKING:
    from pytools.arrays import A1

    from cheartpy.mesh import CheartMeshPatch

    from ._types import ElemSearchMap


class BoundaryRelation(enum.Enum):
    """Boundary relation types."""

    NONE = 0
    SURFACE = 1
    INTERIOR = 2


def classify_boundary_patch[I: np.integer](
    search_map: ElemSearchMap, patch: CheartMeshPatch[I]
) -> A1[I]:
    matchs = np.asarray(
        [len(set[int].intersection(*(search_map[n] for n in elem))) for elem in patch.v],
        dtype=patch.v.dtype,
    )
    kinds = np.unique(matchs)
    log = get_logger()
    log.info("Boundary Type:", tag=patch.tag, number_of_types=len(kinds), types=kinds.tolist())
    return matchs
