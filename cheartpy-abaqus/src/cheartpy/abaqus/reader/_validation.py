from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from ._types import AbaqusMesh, Element


def validate_space_dimension(mesh: AbaqusMesh[Any, Any]) -> bool:
    dims = {v.shape[0] for v in mesh.nodes.v.values()}
    return len(dims) == 1


def validate_element_dimension(elem: Element[np.integer]) -> bool:
    dims = {v.shape[0] for v in elem.v.values()}
    return len(dims) == 1


def validate_element_dimensions(mesh: AbaqusMesh[Any, Any]) -> bool:
    return all(validate_element_dimension(elem) for elem in mesh.elements.values())
