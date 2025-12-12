from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.io.api import fix_ch_sfx
from cheartpy.vtk.api import guess_elem_type_from_dim
from pytools.result import Err, Ok

from .struct import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)

if TYPE_CHECKING:
    from cheartpy.vtk.types import VtkElem, VtkType
    from pytools.arrays import A2


__all__ = ["import_cheart_mesh"]


def _create_bnd_surf[T: np.integer](v: A2[T], tag: int, kind: VtkType) -> CheartMeshPatch[T]:
    bnd = v[v[:, -1] == tag, :-1]
    elems = bnd[:, 0] - 1
    nodes = bnd[:, 1:] - 1
    return CheartMeshPatch(tag, len(bnd), elems, nodes, kind)


def _create_cheart_mesh_surf_from_raw[T: np.integer](
    raw_bnd: A2[T] | None,
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
) -> Ok[CheartMesh[F, I]] | Err:
    prefix = fix_ch_sfx(str(name))
    raw_space = np.loadtxt(f"{prefix}X", dtype=ftype, skiprows=1)
    raw_top = np.loadtxt(f"{prefix}T", dtype=itype, skiprows=1) - 1
    edim = raw_top.shape[1]
    if Path(f"{prefix}B").is_file():
        raw_bnd = np.loadtxt(f"{prefix}B", dtype=itype, skiprows=1)
        bdim = raw_bnd.shape[1] - 2
    else:
        raw_bnd, bdim = None, None
    if forced_type is None:
        match guess_elem_type_from_dim(edim, bdim):
            case Ok(forced_type):
                pass
            case Err(e):
                return Err(e)
    space = CheartMeshSpace(len(raw_space), raw_space)
    top = CheartMeshTopology(len(raw_top), raw_top, forced_type.body)
    bnd = _create_cheart_mesh_surf_from_raw(raw_bnd, forced_type.surf)
    return Ok(CheartMesh(space, top, bnd))
