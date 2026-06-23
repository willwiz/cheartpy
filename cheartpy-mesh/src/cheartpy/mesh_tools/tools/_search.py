from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING

import numpy as np
from pytools.arrays import ToIndex
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from pytools.arrays import A1

type ElemSearchMap = Mapping[ToIndex, set[int]]
type IndexUpdateMap = Mapping[ToIndex, int]


def build_index_update_map[I: np.integer](elements: Mapping[int, A1[I]]) -> IndexUpdateMap:
    """Create a mapping from old node to new continuous node numbering.

    Parameters
    ----------
    elements : Mapping[int, A1[I]]
        The elements to build the index update map from.

    Returns
    -------
    IndexUpdateMap
        A mapping from old node index to new node index.
        Mapping[int, int] where the key is the old node index and the value is the new node index.

    """
    unique_nodes = {n for nodes in elements.values() for n in nodes}
    return {n: i for i, n in enumerate(unique_nodes)}


def build_element_searchmap[I: np.integer](elements: Mapping[int, A1[I]]) -> ElemSearchMap:
    """Create a mapping that returns the elements containing a given node.

    Parameters
    ----------
    elements : Mapping[int, A1[I]]
        The elements to build the search map from.

    Returns
    -------
    ElemSearchMap
        A mapping from node to the elements that contain it.
        Mapping[int, set[int]] where the key is the node and the value is a set of element indices.

    """
    search_map = defaultdict[ToIndex, set[int]](set)
    for elem, nodes in elements.items():
        for node in nodes:
            search_map[node].add(elem)
    return search_map


def search_element(search_map: ElemSearchMap, node: Collection[int]) -> Result[Collection[int]]:
    """Find elements that contain all of the given node."""
    possible_elems = set[int].intersection(*(search_map[n] for n in node))
    if not possible_elems:
        msg = f"No element contains all nodes {node}."
        return Err(ValueError(msg))
    return Ok(possible_elems)
