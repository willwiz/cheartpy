import enum
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from pytools.arrays import A1, A2

type ElemSearchMap = Mapping[int, set[int]]


class GmshBoundaryType(enum.IntEnum):
    NONE = enum.auto()
    SURF = enum.auto()
    INTERNAL = enum.auto()


def build_element_searchmap[I: np.integer](k: A1[I], conn: A2[I]) -> ElemSearchMap:
    """Create a mapping to find elements that contain a given node."""
    search_map = defaultdict[int, set[int]](set)
    for elem, nodes in zip(k, conn, strict=True):
        for node in nodes:
            search_map[node].add(int(elem))
    return search_map


def search_element_association(search_map: ElemSearchMap, nodes: Iterable[int]) -> GmshBoundaryType:
    """Find elements that contain all of the given node."""
    possible_elems = set[int].intersection(*(search_map[n] for n in nodes))
    if not possible_elems:
        return GmshBoundaryType.NONE
    if len(possible_elems) == 1:
        return GmshBoundaryType.SURF
    return GmshBoundaryType.INTERNAL


def find_elem_for_boundary(
    search_map: ElemSearchMap, nodes: Iterable[int], *, strict: bool = True
) -> Result[int]:
    """Find elements that contain all of the given node."""
    possible_elems = set[int].intersection(*(search_map[n] for n in nodes))
    if not possible_elems:
        return Err(ValueError(f"No element found for nodes {nodes}."))
    if len(possible_elems) == 1:
        return Ok(possible_elems.pop())
    if strict:
        return Err(ValueError(f"Multiple elements found for nodes {nodes}: {possible_elems}."))
    return Ok(possible_elems.pop())
