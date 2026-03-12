import operator
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Err, Ok, all_ok

from ._reader import import_abaqus_file
from ._types import AbaqusMesh, Element, Headings, Nodes

if TYPE_CHECKING:
    from pytools.arrays import DType


def nodes_consistent[F: np.floating](a: Nodes[F], b: Nodes[F]) -> bool:
    if not a.v or not b.v:
        return True
    return a == b


def merge_abaqus_meshes[F: np.floating, I: np.integer](
    *meshes: AbaqusMesh[F, I],
) -> Ok[AbaqusMesh[F, I]] | Err:
    if not meshes:
        return Err(ValueError("No meshes provided for merging."))
    if len(meshes) == 1:
        return Ok(meshes[0])
    mesh = meshes[0]
    if not all(nodes_consistent(m.nodes, mesh.nodes) for m in meshes):
        return Err(ValueError("Node definitions are inconsistent across the provided meshes."))
    elements: dict[str, Element[I]] = {}
    for mesh in meshes:
        for name, elem in mesh.elements.items():
            if name not in elements:
                elements[name] = elem
            elif elements[name] != elem:
                msg = (
                    f"Element '{name}' is defined multiple times but is not the same: "
                    f"{elements[name].type} and {elem.type}."
                    f"Please check the Abaqus mesh files for consistency."
                )
                return Err(ValueError(msg))
    headings = Headings(reduce(operator.add, [m.headings for m in meshes]))
    nset = reduce(operator.__ior__, [m.nset for m in meshes])
    elset = reduce(operator.__ior__, [m.elsets for m in meshes])
    return Ok(AbaqusMesh(headings, mesh.nodes, nset, elements, elset, mesh.ftype, mesh.dtype))


def import_abaqus_files[F: np.floating, I: np.integer](
    *files: str,
    ftype: DType[F] = np.float64,
    itype: DType[I] = np.intc,
) -> Ok[AbaqusMesh[F, I]] | Err:
    match all_ok([import_abaqus_file(Path(f), ftype=ftype, dtype=itype) for f in files]):
        case Ok(abaqus_meshes):
            return merge_abaqus_meshes(*abaqus_meshes)
        case Err(e):
            return Err(e)
