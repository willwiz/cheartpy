from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import get_abaqus_boundary_element
from pytools.result import Err, Ok, Result, all_ok

from ._types import ElemIntermediate, ElemSearchMap, Mask
from ._utils import build_element_searchmap, compile_abaqus_elements, search_element

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from cheartpy.abaqus.reader import AbaqusMesh
    from pytools.arrays import A1


def _compile_boundary_patch[F: np.floating, I: np.integer](
    mesh: AbaqusMesh[F, I],
    top: ElemIntermediate[I],
    search_map: ElemSearchMap,
    selections: Iterable[str],
) -> Result[ElemIntermediate[I]]:
    if (surf_type := get_abaqus_boundary_element(top.type)) is None:
        msg = f"Element type '{top.type}' is not supported as a boundary element."
        return Err(ValueError(msg))
    match compile_abaqus_elements(mesh, selections):
        case Ok(elements): ...  # fmt: skip
        case Err(e):
            return Err(e)
    match all_ok([search_element(search_map, nodes) for nodes in elements.v.values()]):
        case Ok(top_id): ...  # fmt: skip
        case Err(e):
            return Err(e)
    return Ok(
        ElemIntermediate(type=surf_type, v=dict(zip(top_id, elements.v.values(), strict=True)))
    )


def compile_boundary_patches[F: np.floating, I: np.integer](
    mesh: AbaqusMesh[F, I],
    top: ElemIntermediate[I],
    boundary: Mapping[int, Sequence[str]] | None,
) -> Ok[Mapping[int, ElemIntermediate[I]]] | Err:
    if boundary is None:
        return Ok({})
    search_map = build_element_searchmap(top)
    return all_ok(
        {k: _compile_boundary_patch(mesh, top, search_map, v) for k, v in boundary.items()}
    )


def define_masks(args: Mapping[str, tuple[str, Sequence[str]]]) -> Mapping[str, Mask]:
    return {k: Mask(k, v[0], v[1]) for k, v in args.items()}


def _create_mask_data[F: np.floating, I: np.integer](
    mesh: AbaqusMesh[F, I], mask: Mask
) -> A1[np.str_]:
    data = np.full(len(mesh.nodes.v), "0", dtype="<U12")
    for elem in (mesh.elements[s] for s in mask.elems):
        for vals in elem.v.values():
            for v in vals:
                data[v] = mask.value
    return data


def compile_mask_data[F: np.floating, I: np.integer](
    mesh: AbaqusMesh[F, I], masks: Mapping[str, tuple[str, Sequence[str]]] | None
) -> dict[str, A1[np.str_]]:
    if masks is None:
        return {}
    _masks = define_masks(masks)
    return {name: _create_mask_data(mesh, m) for name, m in _masks.items()}
