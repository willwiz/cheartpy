__all__ = ["import_cheart_mesh"]
import os
import numpy as np
from ..var_types import *
from .elements import *
from .data import *
from .io import fix_suffix


def _create_cheart_mesh_surf_from_raw(
    raw_bnd: Mat[int_t] | None, surf_type: VtkType | None
) -> CheartMeshBoundary | None:
    if raw_bnd is None or surf_type is None:
        return None
    bnd_tags = np.unique(raw_bnd[:, -1])
    bnd = {tag: create_bnd_surf(raw_bnd, tag) for tag in bnd_tags}
    return CheartMeshBoundary(len(raw_bnd), bnd, surf_type)


def import_cheart_mesh(name: str, force_type: VtkType | None = None) -> CheartMesh:
    prefix = fix_suffix(name)
    raw_space = np.loadtxt(f"{prefix}X", dtype=float, skiprows=1)
    raw_top = np.loadtxt(f"{prefix}T", dtype=int, skiprows=1) - 1
    edim = raw_top.shape[1]
    if os.path.isfile(f"{prefix}B"):
        raw_bnd = np.loadtxt(f"{prefix}B", dtype=int, skiprows=1)
        bdim = raw_bnd.shape[1] - 2
    else:
        raw_bnd = None
        bdim = None
    if force_type is not None:
        vol_type = force_type
        surf_type = None if force_type.surf is None else VTK_ELEM[force_type.surf]
    else:
        vol_type, surf_type = guess_elem_type_from_dim(edim, bdim)
    space = CheartMeshSpace(len(raw_space), raw_space)
    top = CheartMeshTopology(len(raw_top), raw_top, vol_type)
    bnd = _create_cheart_mesh_surf_from_raw(raw_bnd, surf_type)
    return CheartMesh(space, top, bnd)
