from collections import ChainMap, defaultdict
from typing import TYPE_CHECKING

import numpy as np
from pytools.logging import get_logger
from pytools.result import Err, Ok, Result, all_ok

from ._types import ElemIntermediate, ElemSearchMap, IndexUpdateMap

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cheartpy.abaqus.reader import AbaqusMesh
    from cheartpy.elem_interfaces import AbaqusEnum
    from cheartpy.mesh.struct import Mapping
    from pytools.arrays import A1, ToInt


def compile_new_node_map[I: np.integer](elements: ElemIntermediate[I]) -> IndexUpdateMap:
    """Create a mapping from old node to new continuous node numbering."""
    log = get_logger()
    unique_nodes = {n for nodes in elements.v.values() for n in nodes}
    log.debug(
        "Compiling new node map.",
        f"{len(elements.v)} elements found with {len(unique_nodes)} unique nodes.",
    )
    return {n: i for i, n in enumerate(unique_nodes)}


def build_element_searchmap[I: np.integer](elements: ElemIntermediate[I]) -> ElemSearchMap:
    """Create a mapping to find elements that contain a given node."""
    search_map = defaultdict(set)
    for elem, nodes in enumerate(elements.v.values()):
        for node in nodes:
            search_map[node].add(elem)
    return search_map


def search_element(search_map: ElemSearchMap, node: Iterable[ToInt]) -> Result[int]:
    """Find elements that contain all of the given node."""
    possible_elems: set[int]
    possible_elems = set.intersection(*(search_map[n] for n in node))
    if not possible_elems:
        return Err(ValueError(f"No element contains all nodes {node}."))
    if len(possible_elems) > 1:
        return Err(ValueError(f"Multiple elements contain all nodes {node}: {possible_elems}."))
    return Ok(possible_elems.pop())


def merge_abaques_elements[F: np.floating, I: np.integer](
    mesh: AbaqusMesh[F, I],
) -> tuple[Mapping[ToInt, A1[I]], Mapping[ToInt, AbaqusEnum]]:
    elements = ChainMap(*[m.v for m in mesh.elements.values()])
    types = ChainMap(*[dict.fromkeys(m.v, m.type) for m in mesh.elements.values()])
    return elements, types


def find_element_from_elset[F: np.floating, I: np.integer](
    mesh: AbaqusMesh[F, I], selection: str
) -> Result[ElemIntermediate[I]]:
    master, types = merge_abaques_elements(mesh)
    elset = mesh.elsets[selection].v
    type_set = {types[e] for e in elset}
    if len(type_set) != 1:
        msg = (
            f"Elset '{selection}' contains elements of multiple types: {type_set}. "
            f"Please check the Abaqus mesh file for consistency."
        )
        return Err(ValueError(msg))
    elements: dict[ToInt, A1[I]] = {e: master[e] for e in elset}
    if not elements:
        return Err(ValueError(f"Elset '{selection}' does not contain any elements."))
    return Ok(ElemIntermediate(type_set.pop(), elements))


def find_elements[F: np.floating, I: np.integer](
    mesh: AbaqusMesh[F, I], selection: str
) -> Result[ElemIntermediate[I]]:
    if selection in mesh.elements:
        elem = mesh.elements[selection]
        return Ok(ElemIntermediate(type=elem.type, v=elem.v))
    if selection in mesh.elsets:
        return find_element_from_elset(mesh, selection).next()
    msg = f"Topology '{selection}' is not defined in the elements or elsets."
    return Err(ValueError(msg))


def compile_abaqus_elements[F: np.floating, I: np.integer](
    mesh: AbaqusMesh[F, I],
    selections: Iterable[str],
) -> Result[ElemIntermediate[I]]:
    match all_ok([find_elements(mesh, s) for s in selections]):
        case Ok(elements): ...  # fmt: skip
        case Err(e):
            return Err(e)
    element_type = {e.type for e in elements}
    if len(element_type) != 1:
        msg = (
            f"Selected topologies contain elements of multiple types: {element_type}.\n"
            f"Please check the Abaqus mesh file for consistency."
        )
        return Err(ValueError(msg))
    return Ok(ElemIntermediate(type=element_type.pop(), v=ChainMap(*[e.v for e in elements])))
