__all__ = ["import_cheart_mesh"]
import os
import numpy as np
from ..var_types import *
from .elements import *
from .data import *
from .io import fix_suffix


def _create_cheart_mesh_surf_from_raw(raw_bnd: Mat[i32], surf_type: VtkType):
    bnd_tags = np.unique(raw_bnd[:, -1])
    bnd = {tag: create_bnd_surf(raw_bnd, tag) for tag in bnd_tags}
    return _CheartMeshBoundary(len(raw_bnd), bnd, surf_type)


def import_cheart_mesh(name: str) -> CheartMesh:
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
    vol_type, surf_type = guess_elem_type_from_dim(edim, bdim)
    space = _CheartMeshSpace(len(raw_space), raw_space)
    top = _CheartMeshTopology(len(raw_top), raw_top, vol_type)
    bnd = (
        None
        if raw_bnd is None
        else _create_cheart_mesh_surf_from_raw(raw_bnd, surf_type)
    )
    return CheartMesh(space, top, bnd)
