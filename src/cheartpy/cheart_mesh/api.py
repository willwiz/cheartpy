from __future__ import annotations

from pathlib import Path

__all__ = ["import_cheart_mesh"]

from typing import TYPE_CHECKING

import numpy as np

from cheartpy.vtk.api import guess_elem_type_from_dim

from .data import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from .io import fix_suffix

if TYPE_CHECKING:
    from arraystubs import Arr2

    from cheartpy.vtk.trait import VtkElem, VtkType


def create_bnd_surf[T: np.integer](v: Arr2[T], tag: int) -> CheartMeshPatch[T]:
    bnd = v[v[:, -1] == tag, :-1]
    elems = bnd[:, 0] - 1
    nodes = bnd[:, 1:] - 1
    return CheartMeshPatch(tag, len(bnd), elems, nodes)


def _create_cheart_mesh_surf_from_raw[T: np.integer](
    raw_bnd: Arr2[T] | None,
    surf_type: VtkType | None,
) -> CheartMeshBoundary[T] | None:
    if raw_bnd is None or surf_type is None:
        return None
    bnd_tags = np.unique(raw_bnd[:, -1])
    bnd = {int(tag): create_bnd_surf(raw_bnd, int(tag)) for tag in bnd_tags}
    return CheartMeshBoundary(len(raw_bnd), bnd, surf_type)


def import_cheart_mesh(
    name: Path | str,
    forced_type: VtkElem | None = None,
) -> CheartMesh[np.float64, np.intc]:
    prefix = fix_suffix(str(name))
    raw_space = np.loadtxt(f"{prefix}X", dtype=np.float64, skiprows=1)
    raw_top = np.loadtxt(f"{prefix}T", dtype=np.intc, skiprows=1) - 1
    edim = raw_top.shape[1]
    if Path(f"{prefix}B").exists():
        raw_bnd = np.loadtxt(f"{prefix}B", dtype=int, skiprows=1)
        bdim = raw_bnd.shape[1] - 2
    else:
        raw_bnd, bdim = None, None
    if forced_type is None:
        forced_type = guess_elem_type_from_dim(edim, bdim)
    space = CheartMeshSpace(len(raw_space), raw_space)
    top = CheartMeshTopology(len(raw_top), raw_top, forced_type.elem)
    bnd = _create_cheart_mesh_surf_from_raw(raw_bnd, forced_type.surf)
    return CheartMesh(space, top, bnd)
