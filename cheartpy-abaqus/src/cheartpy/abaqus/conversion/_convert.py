from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import get_cheart_order_for_abaqus, get_vtk_element_for_abaqus
from cheartpy.mesh.struct import (
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from pytools.result import Err, Ok, Result, all_ok

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cheartpy.abaqus.reader import AbaqusMesh
    from pytools.arrays import A1

    from ._types import ElemIntermediate, IndexUpdateMap


def create_mesh_space[F: np.floating, I: np.integer](
    mesh: AbaqusMesh[F, I],
    nmap: IndexUpdateMap,
) -> CheartMeshSpace[F]:
    dtype = next(iter(mesh.nodes.v.values())).dtype
    return CheartMeshSpace(
        n=len(nmap),
        v=np.ascontiguousarray([mesh.nodes.v[k] for k in nmap], dtype=dtype),
    )


def create_mesh_topology[I: np.integer](
    top: ElemIntermediate[I],
    nmap: IndexUpdateMap,
) -> Result[CheartMeshTopology[I]]:
    dtype = next(iter(top.v.values())).dtype
    if (kind := get_vtk_element_for_abaqus(top.type)) is None:
        msg = f"Element type '{top.type}' is not supported."
        return Err(ValueError(msg))
    order = get_cheart_order_for_abaqus(top.type)
    if len(order) != len(next(iter(top.v.values()))):
        msg = f"Element type '{top.type}' has an unexpected number of nodes."
        return Err(ValueError(msg))
    data = np.array([[nmap[e[n]] for n in order] for e in top.v.values()], dtype=dtype)
    return Ok(CheartMeshTopology(n=len(data), v=np.ascontiguousarray(data), TYPE=kind))


def create_mesh_boundary_patch[I: np.integer](
    top: ElemIntermediate[I],
    nmap: IndexUpdateMap,
    tag: int,
    bc_patch: ElemIntermediate[I],
) -> Result[CheartMeshPatch[I]]:
    dtype = next(iter(top.v.values())).dtype
    elems = np.ascontiguousarray(list(bc_patch.v.keys()), dtype=dtype)
    if (kind := get_vtk_element_for_abaqus(bc_patch.type)) is None:
        msg = f"Boundary type '{top.type}' is not supported."
        return Err(ValueError(msg))
    order = get_cheart_order_for_abaqus(bc_patch.type)
    if len(order) != len(next(iter(bc_patch.v.values()))):
        msg = f"Boundary type '{top.type}' has an unexpected number of nodes."
        return Err(ValueError(msg))
    data = np.array([[nmap[p[n]] for n in order] for p in bc_patch.v.values()], dtype=dtype)
    return Ok(CheartMeshPatch(tag=tag, n=len(data), k=elems, v=data, TYPE=kind))


def create_mesh_boundary[I: np.integer](
    top: ElemIntermediate[I],
    nmap: IndexUpdateMap,
    boundary: Mapping[int, ElemIntermediate[I]],
) -> Ok[CheartMeshBoundary[I]] | Ok[None] | Err:
    if not boundary:
        return Ok(None)
    match all_ok({k: create_mesh_boundary_patch(top, nmap, k, v) for k, v in boundary.items()}):
        case Ok(patches): ...  # fmt: skip
        case Err(e):
            return Err(e)
    types = {p.TYPE for p in patches.values()}
    if len(types) != 1:
        msg = f"Boundary patches have different types, cannot merge them: {types}"
        return Err(ValueError(msg))
    return Ok(CheartMeshBoundary(len(patches), v=patches, TYPE=types.pop()))


def create_mesh_masks[F: np.floating, I: np.integer](
    nmap: IndexUpdateMap, masks: Mapping[str, A1[np.str_]] | None
) -> Result[Mapping[str, A1[np.str_]]]:
    if masks is None:
        return Ok({})
    return Ok({k: np.array([mask[i] for i in nmap]) for k, mask in masks.items()})
