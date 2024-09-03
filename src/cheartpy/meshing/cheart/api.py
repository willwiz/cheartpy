import os
from .data import *


def import_cheart_mesh(name: str) -> CheartMesh:
    prefix = fix_suffix(name)
    raw_space = np.loadtxt(f"{prefix}X", dtype=float, skiprows=1)
    raw_top = np.loadtxt(f"{prefix}T", dtype=int, skiprows=1) - 1
    space = CheartMeshSpace(len(raw_space), raw_space)
    top = CheartMeshTopology(len(raw_top), raw_top)
    if os.path.isfile(f"{prefix}B"):
        raw_bnd = np.loadtxt(f"{prefix}B", dtype=int, skiprows=1)
        bnd_tags = np.unique(raw_bnd[:, -1])
        bnd = CheartMeshBoundary(
            len(raw_bnd), {tag: create_bnd_surf(raw_bnd, tag) for tag in bnd_tags}
        )
    else:
        bnd = None
    return CheartMesh(space, top, bnd)
