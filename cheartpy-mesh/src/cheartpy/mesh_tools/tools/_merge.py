from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Ok, Result

from cheartpy.mesh import CheartMesh, CheartMeshSpace, CheartMeshTopology

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pytools.arrays import A1, A2


def merge_meshes[F: np.floating, I: np.integer](
    meshes: Sequence[CheartMesh[F, I]], vs: Mapping[str, Sequence[A2[F]]]
) -> Result[tuple[CheartMesh[F, I], CheartMesh[F, I], Mapping[str, A2[F]]]]:
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
    return Ok((merged_mesh, interface_mesh, {k: np.concatenate(v, axis=0) for k, v in vs.items()}))
