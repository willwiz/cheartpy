from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Ok, Result

from cheartpy.mesh import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)

from ._search import build_index_update_map
from ._types import MergedMesh

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pytools.arrays import A1, A2, ToInt


def merge_meshes[F: np.floating, I: np.integer](
    meshes: Sequence[CheartMesh[F, I]], vs: Mapping[str, Sequence[A2[F]]]
) -> Result[MergedMesh[F, I]]:
    """Merge Cheart meshes into one, and combine variable field if given.

    Parameters
    ----------
    meshes
        The meshes to merge. Must have the same topology type and space dtype.
    vs
        The variable fields to merge.

    Returns [Result object]
    -----------------------
    CheartMesh[F, I]
        Merged Volume Mesh
    CheartMesh[F, I]
        Interface to identify original meshes
    Mapping[str, A2[F]]]]
        Merged variable fields, if given. The shape of each field is (n_nodes, n_components).

    """
    ftype = meshes[0].space.v.dtype
    dtype = meshes[0].top.v.dtype
    mesh_sizes = [0] + [int(m.space.n) for m in meshes]
    node_offset: A1[I] = np.add.accumulate(mesh_sizes)
    merged_space = np.zeros((node_offset[-1], 3), dtype=ftype)
    for m, offset in zip(meshes, node_offset, strict=False):
        merged_space[offset : offset + m.space.n] = m.space.v
    tops: list[A2[I]] = [
        (m.top.v + offset).astype(dtype) for m, offset in zip(meshes, node_offset, strict=False)
    ]
    merged_top = np.concatenate(tops, axis=0)
    merged_mesh = CheartMesh(
        space=CheartMeshSpace(n=node_offset[-1], v=merged_space),
        top=CheartMeshTopology(n=len(merged_top), v=merged_top, TYPE=meshes[0].top.TYPE),
        bnd=None,
    )

    interface_space = CheartMeshSpace(
        n=len(meshes), v=np.arange(len(meshes), dtype=ftype).reshape(-1, 1)
    )
    elem_map = [np.ones((m.top.n, 1), dtype=dtype) * i for i, m in enumerate(meshes)]
    interface_mesh = CheartMesh(
        space=interface_space,
        top=CheartMeshTopology(
            n=sum(m.top.n for m in meshes),
            v=np.concatenate(elem_map, axis=0).astype(dtype),
            TYPE=meshes[0].top.TYPE,
        ),
        bnd=None,
    )
    return Ok(
        MergedMesh(
            merged_mesh, interface_mesh, {k: np.concatenate(v, axis=0) for k, v in vs.items()}
        )
    )

def _create_new_bnd[T: np.integer](
    p: CheartMeshPatch()[T],
    node_map: Mapping[T, ToInt],
) -> CheartMeshPatch[T]:
    new_v = np.array([[node_map[i] for i in patch] for patch in p.v], dtype=int)
    return CheartMeshPatch(p.tag, p.n, p.k, new_v, p.TYPE)


def recompile_cheart_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> CheartMesh[F, I]:
    """Recompile a Cheart mesh to ensure that the node indices are contiguous and start from 0.

    Parameters
    ----------
    mesh
        The Cheart mesh to recompile.

    Returns [Result object]
    -----------------------
    CheartMesh[F, I]
        Recompiled Cheart mesh.

    """
    node_map = build_index_update_map(mesh.top.v)
    new_x = mesh.space.v[list(node_map.keys())]
    new_t = np.ascontiguousarray(
        [[node_map[i] for i in elem] for elem in mesh.top.v], dtype=mesh.top.v.dtype
    )
    boundary = (
        CheartMeshBoundary(
            mesh.bnd.n,
            {k: _create_new_bnd(v, node_map) for k, v in mesh.bnd.v.items()},
            mesh.bnd.TYPE,
        )
        if mesh.bnd
        else None
    )
    return CheartMesh(
        CheartMeshSpace(len(new_x), new_x),
        CheartMeshTopology(len(new_t), new_t, mesh.top.TYPE),
        boundary,
    )
