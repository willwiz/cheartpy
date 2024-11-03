from collections import defaultdict
from typing import Mapping, Sequence
from ...var_types import *
from ...tools.basiclogging import _Logger, NullLogger


def create_node2elem_map(top: Mat[i32]) -> Mapping[int, Sequence[int]]:
    """
    Given the elems which has corresponding node
    """
    n2e_map: dict[int, list[int]] = defaultdict(list)
    for i, elem in enumerate(top):
        for node in elem:
            n2e_map[node].append(i)
    return n2e_map


def get_elem_neighbour(map: Mapping[int, Sequence[int]], top: Mat[i32], i: int):
    """
    Count the number of neighbours (not unique)
    Return values:
    0: element is disconnected from the rest
    1: element shares 1 node
    2: element shares an edge or connects to 2 element
    >2: element is okay

    0 and 1 are not okay
    """
    return [e for n in top[i] for e in map[n] if e != i]


def create_elem_connectivity(map: Mapping[int, Sequence[int]], top: Mat[i32]):
    e2e_map: dict[int, list[int]] = dict()
    for i in range(len(top)):
        e2e_map[i] = [k for k in set(get_elem_neighbour(map, top, i))]
    return e2e_map


def get_connected_subset(
    e2e_map: Mapping[int, Sequence[int]], elem: int, elem_set: set[int] | None = None
) -> set[int]:
    elem_set = set() if elem_set is None else elem_set
    connected_elems = e2e_map[elem]
    new_elems = {i for i in connected_elems if not i in elem_set}
    updated_set = elem_set | new_elems
    for i in new_elems:
        updated_set = updated_set | get_connected_subset(e2e_map, i, updated_set)
    return updated_set


def get_connected_subsets(
    e2e_map: Mapping[int, Sequence[int]], LOG: _Logger = NullLogger()
):
    subsets: list[set[int]] = list()
    current_set = set(e2e_map.keys())
    while current_set:
        new_set = get_connected_subset(e2e_map, next(iter(current_set)))
        subsets.append(new_set)
        LOG.debug(f"{new_set}")
        current_set = current_set - new_set
        LOG.debug(f"{current_set}")
    return subsets
