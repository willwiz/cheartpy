__all__ = ["import_cheart_mesh"]

from pathlib import Path

import numpy as np
from arraystubs import Arr2
from cheartpy.io.api import fix_suffix
from cheartpy.vtk.api import guess_elem_type_from_dim
from cheartpy.vtk.trait import VtkElem, VtkType

from .struct import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)


def _create_bnd_surf[T: np.integer](v: Arr2[T], tag: int, kind: VtkType) -> CheartMeshPatch[T]:
    bnd = v[v[:, -1] == tag, :-1]
    elems = bnd[:, 0] - 1
    nodes = bnd[:, 1:] - 1
    return CheartMeshPatch(tag, len(bnd), elems, nodes, kind)


def _create_cheart_mesh_surf_from_raw[T: np.integer](
    raw_bnd: Arr2[T] | None,
    surf_type: VtkType | None,
) -> CheartMeshBoundary[T] | None:
    if raw_bnd is None or surf_type is None:
        return None
    bnd_tags = np.unique(raw_bnd[:, -1])
    bnd = {int(tag): _create_bnd_surf(raw_bnd, int(tag), surf_type) for tag in bnd_tags}
    return CheartMeshBoundary(len(raw_bnd), bnd, surf_type)


def import_cheart_mesh[F: np.floating, I: np.integer](
    name: Path | str,
    forced_type: VtkElem | None = None,
    *,
    ftype: type[F] = np.float64,
    itype: type[I] = np.intc,
) -> CheartMesh[F, I]:
    prefix = fix_suffix(str(name))
    raw_space = np.loadtxt(f"{prefix}X", dtype=ftype, skiprows=1)
    raw_top = np.loadtxt(f"{prefix}T", dtype=itype, skiprows=1) - 1
    edim = raw_top.shape[1]
    if Path(f"{prefix}B").exists():
        raw_bnd = np.loadtxt(f"{prefix}B", dtype=itype, skiprows=1)
        bdim = raw_bnd.shape[1] - 2
    else:
        raw_bnd, bdim = None, None
    if forced_type is None:
        forced_type = guess_elem_type_from_dim(edim, bdim)
    space = CheartMeshSpace(len(raw_space), raw_space)
    top = CheartMeshTopology(len(raw_top), raw_top, forced_type.body)
    bnd = _create_cheart_mesh_surf_from_raw(raw_bnd, forced_type.surf)
    return CheartMesh(space, top, bnd)
