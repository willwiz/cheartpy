import enum
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from pytools.logging import get_logger

if TYPE_CHECKING:
    from pytools.arrays import A1

    from cheartpy.mesh import CheartMeshPatch

    from ._sparse_search import ElementSparseMap
    from ._types import ElemSearchMap


class BoundaryRelation(enum.Enum):
    """Boundary relation types."""

    NONE = 0
    SURFACE = 1
    INTERIOR = 2


def classify_boundary_patch[I: np.integer](
    search_map: ElemSearchMap, patch: CheartMeshPatch[I]
) -> A1[I]:
    t0 = perf_counter()
    matchs = np.asarray(
        [len(set[int].intersection(*(search_map[n] for n in elem))) for elem in patch.v],
        dtype=patch.v.dtype,
    )
    t1 = perf_counter()
    kinds = np.unique(matchs)
    log = get_logger()
    log.info(
        "Boundary Type:",
        tag=patch.tag,
        number_of_types=len(kinds),
        types=kinds.tolist(),
        time=t1 - t0,
    )
    return matchs


def classify_boundary_patch_sparse[I: np.integer](
    search_map: ElementSparseMap[I], patch: CheartMeshPatch[I]
) -> A1[I]:
    t0 = perf_counter()
    matchs = search_map.search_elements(patch.v).classify_element_relation()
    t1 = perf_counter()
    kinds = np.unique(matchs)
    log = get_logger()
    log.info(
        "Boundary Type (Sparse):",
        tag=patch.tag,
        number_of_types=len(kinds),
        types=kinds.tolist(),
        time=t1 - t0,
    )
    return matchs
