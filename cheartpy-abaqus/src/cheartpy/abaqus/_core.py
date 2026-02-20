import operator as op
from collections import defaultdict
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.io.api import chwrite_str_utf
from cheartpy.mesh.struct import (
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from pytools.result import Err, Ok, all_ok

from ._impl import get_vtktype_from_abaqus_enum
from ._trait import AbaqusElement

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pytools.arrays import A1, A2

    from ._struct import Mask, MeshElements, MeshNodes


def get_abaqus_element(tag: str, dim: int) -> Ok[AbaqusElement] | Err:
    match tag, dim:
        case "T3D2", 2:
            kind = AbaqusElement.T3D2
        case "T3D3", 3:
            kind = AbaqusElement.T3D3
        case "CPS3", 3:
            kind = AbaqusElement.CPS3
        case "CPS4", 4:
            kind = AbaqusElement.CPS4
        case "CPS4", 9:
            kind = AbaqusElement.CPS4_3D
        case "C3D4", 4:
            kind = AbaqusElement.C3D4
        case "S3R", 3:
            kind = AbaqusElement.S3R
        case _, 8:
            kind = AbaqusElement.TetQuad3D
        case _, 4:
            kind = AbaqusElement.Tet3D
        case _:
            msg = f"Element type '{type}' with dimension {dim} is not implemented. "
            return Err(ValueError(msg))
    return Ok(kind)


def check_for_elements[I: np.integer](
    elems: Mapping[str, MeshElements[I]],
    topology: Sequence[str],
    boundary: Mapping[int, Sequence[str]] | None = None,
) -> Ok[None] | Err:
    for name in topology:
        if name not in elems:
            msg = f"Topology '{name}' is not defined in the elements."
            return Err(ValueError(msg))
    if boundary is not None:
        for name in boundary.values():
            if name not in elems:
                msg = f"Boundary '{name}' is not defined in the elements."
                return Err(ValueError(msg))
    return Ok(None)


def build_element_map[I: np.integer](
    elems: Mapping[str, MeshElements[I]],
    topology: Sequence[str],
) -> dict[int, int]:
    unique_nodes = {int(v) for name in topology for elem in elems[name].v for v in elem}
    elmap: dict[int, int] = {}
    return {elmap[node]: i for i, node in enumerate(sorted(unique_nodes), start=1)}


def create_topology[I: np.integer](
    elmap: Mapping[int, int],
    elems: Mapping[str, MeshElements[I]],
    topology: str,
) -> Ok[CheartMeshTopology[I]] | Err:
    data: list[list[int]] = []
    top = elems[topology]
    match get_abaqus_element(top.kind, top.v.shape[1]):
        case Ok(el):
            data.extend([elmap[values[j]] for j in el.value.nodes] for values in top.v)
            kind = get_vtktype_from_abaqus_enum(el)
        case Err(e):
            return Err(e)
    topology_data = np.array(data, dtype=int)
    return Ok(
        CheartMeshTopology(
            n=len(topology_data),
            v=np.ascontiguousarray(topology_data),
            TYPE=kind,
        )
    )


def merge_topologies[I: np.integer](*top: CheartMeshTopology[I]) -> Ok[CheartMeshTopology[I]] | Err:
    types = {t.TYPE for t in top}
    if len(types) != 1:
        msg = f"Topologies have different types, cannot merge them: {types}"
        return Err(ValueError(msg))
    dims = {t.v.shape[1] for t in top}
    if len(dims) != 1:
        msg = f"Topologies have different dimensions: {dims}There most likely is a bug in the code."
        return Err(ValueError(msg))
    n = sum(t.n for t in top)
    v = np.concatenate([t.v for t in top], axis=0)
    if len(v) != n:
        msg = (
            "Topologies have different number of elements, cannot merge them. "
            f"Total number of elements: {n}, but got {len(v)}."
        )
        return Err(ValueError(msg))
    return Ok(
        CheartMeshTopology(
            n=n,
            v=np.ascontiguousarray(v, dtype=int),
            TYPE=types.pop(),
        )
    )


def topology_hashmap[I: np.integer](topology: CheartMeshTopology[I]) -> dict[int, set[int]]:
    """Return a hashmap of nodes and a set of the elements it is part of."""
    hashmap: dict[int, set[int]] = defaultdict(set)
    for i, elem in enumerate(topology.v):
        for node in elem:
            hashmap[int(node)].add(i)
    return hashmap


def find_element_by_nodes[I: np.integer](
    top_hashmap: Mapping[int, set[int]],
    nodes: A1[I],
) -> Ok[int] | Err:
    element_sets = [top_hashmap[int(node)] for node in nodes]
    elements: set[int] = reduce(op.and_, element_sets)
    if len(elements) != 1:
        msg = (
            f"Nodes {nodes} are not unique in the topology"
            f"Found {len(elements)} elements: {elements}. "
            "Please check the Abaqus mesh files for consistency."
        )
        return Err(ValueError(msg))
    return Ok(elements.pop())


def create_boundary_patch[I: np.integer](
    elmap: Mapping[int, int],
    top_hashmap: Mapping[int, set[int]],
    elems: MeshElements[I],
    tag: int,
) -> Ok[CheartMeshPatch[I]] | Err:
    array_dim = elems.v.shape[1]
    match get_abaqus_element(elems.kind, array_dim):
        case Ok(abaqus_elem):
            kind = get_vtktype_from_abaqus_enum(abaqus_elem)
            nodes: A2[I] = np.array(
                [
                    [elmap[int(node)] for node in patch[abaqus_elem.value.nodes]]
                    for patch in elems.v
                ],
                dtype=elems.v.dtype,
            )
        case Err(e):
            return Err(e)
    match all_ok([find_element_by_nodes(top_hashmap, row) for row in nodes]):
        case Ok(vals):
            elements = np.array(vals, dtype=nodes.dtype)
        case Err(e):
            return Err(e)
    return Ok(CheartMeshPatch(tag=tag, n=len(elems.v), k=elements, v=nodes, TYPE=kind))


def merge_boundary_patches[I: np.integer](
    *patches: Ok[CheartMeshPatch[I]] | Err,
) -> Ok[CheartMeshPatch[I]] | Err:
    match all_ok(patches):
        case Ok(ok_patches): ...  # fmt: skip
        case Err(e):
            return Err(e)
    types = {p.TYPE for p in ok_patches}
    if len(types) != 1:
        msg = f"Boundary patches have different types, cannot merge them: {types}"
        return Err(ValueError(msg))
    tags = {p.tag for p in ok_patches}
    if len(tags) != len(ok_patches):
        msg = f"Boundary patches have different tags, cannot merge them: {tags}"
        return Err(ValueError(msg))
    n = sum(p.n for p in ok_patches)
    keys = np.concatenate([p.k for p in ok_patches], axis=0)
    if len(keys) != n:
        msg = (
            "Boundary patches have different number of elements, cannot merge them. "
            f"Total number of elements: {n}, but got {len(keys)}."
        )
        return Err(ValueError(msg))
    values = np.concatenate([p.v for p in ok_patches], axis=0)
    if len(values) != n:
        msg = (
            "Boundary patches have different number of elements, cannot merge them. "
            f"Total number of elements: {n}, but got {len(values)}."
        )
        return Err(ValueError(msg))
    return Ok(CheartMeshPatch(tag=tags.pop(), n=n, k=keys, v=values, TYPE=types.pop()))


def create_boundaries[F: np.floating, I: np.integer](
    elmap: Mapping[int, int],
    top_hashmap: Mapping[int, set[int]],
    elems: Mapping[str, MeshElements[I]],
    boundary: Mapping[int, Sequence[str]] | None,
) -> Ok[CheartMeshBoundary[I]] | Ok[None] | Err:
    if boundary is None:
        return Ok(None)
    patch_exists = {k: k in elems for k in boundary}
    if not all(patch_exists.values()):
        missing = [k for k, v in patch_exists.items() if not v]
        msg = f"Boundary patches {missing} are not defined in the elements."
        return Err(ValueError(msg))
    match all_ok(
        [
            merge_boundary_patches(
                *[create_boundary_patch(elmap, top_hashmap, elems[v], k) for v in b],
            )
            for k, b in boundary.items()
            if k in elems
        ]
    ):
        case Ok(patches): ...  # fmt: skip
        case Err(e):
            return Err(e)
    types = {p.TYPE for p in patches}
    if len(types) != 1:
        msg = f"Boundary patches have different types, cannot merge them: {types}"
        return Err(ValueError(msg))
    return Ok(CheartMeshBoundary(n=len(patches), v={p.tag: p for p in patches}, TYPE=types.pop()))


def create_mask[I: np.integer](
    elmap: Mapping[int, int],
    elems: Mapping[str, MeshElements[I]],
    masks: Mask,
) -> None:
    data = np.full((len(elmap), 1), "0", dtype="<U12")
    for elem in (elems[s] for s in masks.elems):
        for vals in elem.v:
            for v in vals:
                data[elmap[int(v)] - 1] = masks.value
    chwrite_str_utf(masks.name, data)


def create_space[F: np.floating](
    nodes: MeshNodes[F],
    elmap: Mapping[int, int],
) -> CheartMeshSpace[F]:
    return CheartMeshSpace(
        n=len(elmap),
        v=np.ascontiguousarray(nodes.v[list(elmap.keys())], dtype=nodes.v.dtype),
    )
