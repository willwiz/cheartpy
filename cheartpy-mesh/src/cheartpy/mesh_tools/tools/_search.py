from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from pytools.arrays import ToIndex, ToInt
from pytools.result import Err, Ok, Result, all_ok

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from pytools.arrays import A1, A2

    from ._types import ElemSearchMap, IndexUpdateMap


def build_index_update_map[I: np.integer](elements: A2[I]) -> IndexUpdateMap:
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
    unique_nodes = {n for nodes in elements for n in nodes}
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


def search_element(search_map: ElemSearchMap, node: Collection[ToInt]) -> Result[set[int]]:
    """Find elements that contain all of the given node."""
    possible_elems = set[int].intersection(*(search_map[n] for n in node))
    if not possible_elems:
        msg = f"No element contains all nodes {node}."
        return Err(ValueError(msg))
    return Ok(possible_elems)


def search_element_unique(search_map: ElemSearchMap, node: Collection[ToInt]) -> Result[int]:
    """Find the unique element that contains all of the given node."""
    match search_element(search_map, node):
        case Ok(possible_elems): ...  # fmt: skip
        case Err(e):
            return Err(e)
    if len(possible_elems) > 1:
        return Err(ValueError(f"Multiple elements contain all nodes {node}: {possible_elems}."))
    return Ok(possible_elems.pop())


def search_elements_from_boundary_set[I: np.integer](
    elemnts: A2[I], bnd: A2[I]
) -> Result[Sequence[int]]:
    """Find elements that contain all of the nodes in the boundary set."""
    search_map = build_element_searchmap(dict(enumerate(elemnts)))
    return all_ok(search_element_unique(search_map, b) for b in bnd).next()
