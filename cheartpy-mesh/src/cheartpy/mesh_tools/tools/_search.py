from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeIs, overload

import numpy as np
from pytools.result import Err, Ok, Result, all_ok

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from pytools.arrays import A1, A2, ToInt

    from ._types import ElemSearchMap


def build_element_searchmap[I: np.integer](
    connectivity: A2[I], *, keys: A1[I] | None = None
) -> Result[ElemSearchMap]:
    """Create a mapping that returns the elements containing a given node.

    Parameters
    ----------
    connectivity : Mapping[int, A1[I]]
        The elements to build the search map from.

    keys : A1[I] | None, default=np.arange(len(connectivity), dtype=connectivity.dtype)
        The keys to use for the elements. If None, the indices of the elements will be used.

    Returns
    -------
    ElemSearchMap
        A mapping from node to the elements that contain it.
        Mapping[int, set[int]] where the key is the node and the value is a set of element indices.

    """
    if keys is None:
        keys = np.arange(len(connectivity), dtype=connectivity.dtype)
    if len(keys) != len(connectivity):
        msg = f"Keys length {len(keys)} does not match connectivity length {len(connectivity)}."
        return Err(ValueError(msg))
    search_map: ElemSearchMap = defaultdict(set)
    for elem, nodes in zip(keys, connectivity, strict=True):
        for node in nodes:
            search_map[node].add(int(elem))
    return Ok(search_map)


def search_element(search_map: ElemSearchMap, node: Iterable[ToInt]) -> Result[set[int]]:
    """Find elements that contain all of the given node."""
    possible_elems = set[int].intersection(*(search_map[n] for n in node))
    if not possible_elems:
        msg = f"No element contains all nodes {node}."
        return Err(ValueError(msg))
    return Ok(possible_elems)


_1D = 1
_2D = 2


def _is_1d[I: np.generic](arr: np.ndarray[Any, np.dtype[I]]) -> TypeIs[A1[I]]:
    return arr.ndim == _1D


def _is_2d[I: np.generic](arr: np.ndarray[Any, np.dtype[I]]) -> TypeIs[A2[I]]:
    return arr.ndim == _2D


def _get_search_map[I: np.integer](
    top: ElemSearchMap | A2[I], keys: A1[I] | None = None
) -> Result[ElemSearchMap]:
    if isinstance(top, Mapping):
        return Ok(top)
    return build_element_searchmap(top, keys=keys).next()


@overload
def _find_element[I: np.integer](
    search_map: ElemSearchMap, nodes: A1[I], *, unique: Literal[True]
) -> Result[int]: ...
@overload
def _find_element[I: np.integer](
    search_map: ElemSearchMap, nodes: A1[I], *, unique: Literal[False]
) -> Result[Collection[int]]: ...
def _find_element[I: np.integer](
    search_map: ElemSearchMap, nodes: A1[I], *, unique: bool
) -> Result[int] | Result[Collection[int]]:
    match search_element(search_map, nodes):
        case Ok(possible_elems): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    if not unique:
        return Ok(possible_elems)
    if len(possible_elems) > 1:
        msg = f"Multiple elements contain all nodes {nodes}: {possible_elems}."
        return Err(ValueError(msg))
    return Ok(possible_elems.pop())


@overload
def _find_elements[I: np.integer](
    top: ElemSearchMap, nodes: A2[I], *, unique: Literal[False]
) -> Result[Sequence[Collection[int]]]: ...
@overload
def _find_elements[I: np.integer](
    top: ElemSearchMap, nodes: A2[I], *, unique: Literal[True]
) -> Result[Sequence[int]]: ...
def _find_elements[I: np.integer](
    top: ElemSearchMap, nodes: A2[I], *, unique: bool
) -> Result[Sequence[int]] | Result[Sequence[Collection[int]]]:
    match all_ok([search_element(top, n) for n in nodes]):
        case Ok(possible_elems): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    if not unique:
        return Ok(possible_elems)
    if any(len(elems) > 1 for elems in possible_elems):
        msg = f"Multiple elements contain all nodes in one of the rows of {nodes}."
        return Err(ValueError(msg))
    return Ok([elems.pop() for elems in possible_elems])


def find_elements[I: np.integer](
    top: ElemSearchMap | A2[I],
    nodes: np.ndarray[tuple[int], np.dtype[I]] | np.ndarray[tuple[int, int], np.dtype[I]],
    *,
    unique: bool = True,
    keys: A1[I] | None = None,
) -> (
    Result[int]
    | Result[Collection[int]]
    | Result[Sequence[int]]
    | Result[Sequence[Collection[int]]]
):
    match _get_search_map(top, keys=keys):
        case Ok(search_map): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    match nodes:
        case np.ndarray() if _is_1d(nodes):
            return _find_element(search_map, nodes, unique=unique)
        case np.ndarray() if _is_2d(nodes):
            return _find_elements(search_map, nodes, unique=unique)
        case _:
            msg = f"Nodes array must be 1D or 2D, got {nodes.ndim}D."
            return Err(ValueError(msg))
