import dataclasses as dc
import itertools
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import (
    GmshEnum,
)

import gmsh

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cheartpy.gmsh.types import Tag
    from pytools.arrays import A1, A2, DType, ToInt


@dc.dataclass(slots=True)
class GmshElements[I: np.integer]:
    kind: GmshEnum
    e: A1[I]
    conn: A2[I]


def get_element[I: np.integer = np.intp](
    el_type: ToInt, tags: A1[np.integer], nodes: A1[np.integer], *, dtype: DType[I] = np.intp
) -> GmshElements[I]:
    _, _, _, elem_size, _, _ = gmsh.model.mesh.get_element_properties(el_type)
    return GmshElements(
        kind=GmshEnum(el_type),
        e=np.ascontiguousarray(tags, dtype=dtype) - 1,
        conn=np.ascontiguousarray(nodes, dtype=dtype).reshape(-1, elem_size) - 1,
    )


def merge_elements[I: np.integer](elements: Sequence[GmshElements[I]]) -> GmshElements[I]:
    elements = list(elements)

    if not elements:
        msg = "No elements to merge."
        raise ValueError(msg)
    el_types = {el.kind for el in elements}
    if len(el_types) != 1:
        msg = f"Cannot merge elements of different types: {el_types}."
        raise ValueError(msg)
    tags = np.concatenate([el.e for el in elements])
    conn = np.concatenate([el.conn for el in elements])
    return GmshElements(kind=el_types.pop(), e=tags, conn=conn)


def get_gmsh_entity[I: np.integer = np.intp](
    dim: int, entity_tag: Sequence[Tag], *, dtype: DType[I] = np.intp
) -> GmshElements[I]:
    res = list(
        itertools.chain.from_iterable(
            zip(*gmsh.model.mesh.get_elements(dim=dim, tag=i), strict=True) for i in entity_tag
        )
    )
    return merge_elements([get_element(t, tag, nodes, dtype=dtype) for t, tag, nodes in res])


def gmsh_finalize(filename: Path | None = None) -> None:
    """Finalize the GMSH API."""
    if not gmsh.is_initialized():
        print("GMSH API is not initialized. No need to finalize.")
    if filename:
        gmsh.write(str(filename))
    else:
        gmsh.fltk.run()
    gmsh.finalize()
