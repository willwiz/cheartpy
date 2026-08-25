from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from pytools.arrays import A1, A2

type ElemSearchMap = Mapping[int, set[int]]


def build_element_searchmap[I: np.integer](k: A1[I], conn: A2[I]) -> ElemSearchMap:
    """Create a mapping to find elements that contain a given node."""
    search_map = defaultdict[int, set[int]](set)
    for elem, nodes in zip(k, conn, strict=True):
        for node in nodes:
            search_map[node].add(int(elem))
    return search_map


def search_element(search_map: ElemSearchMap, node: Iterable[int]) -> Result[int]:
    """Find elements that contain all of the given node."""
    possible_elems = set[int].intersection(*(search_map[n] for n in node))
    if not possible_elems:
        msg = f"No element contains all nodes {node}."
        return Err(ValueError(msg))
    if len(possible_elems) > 1:
        return Err(ValueError(f"Multiple elements contain all nodes {node}: {possible_elems}."))
    return Ok(possible_elems.pop())
