from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import (
    VtkEnum,
    get_vtk_boundary_element,
    guess_vtk_elem_from_dim,
)
from cheartpy.io import fix_ch_sfx
from pytools.result import Err, Ok, Result

from ._struct import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)

if TYPE_CHECKING:
    from pytools.arrays import A2, DType, ToInt

__all__ = ["import_cheart_mesh"]


def _create_bnd_surf[T: np.integer](v: A2[T], tag: ToInt, kind: VtkEnum) -> CheartMeshPatch[T]:
    bnd = v[v[:, -1] == tag, :-1]
    elems = bnd[:, 0] - 1
    nodes = bnd[:, 1:] - 1
    return CheartMeshPatch(tag, len(bnd), elems, nodes, kind)


def _create_cheart_mesh_surf_from_raw[T: np.integer](
    raw_bnd: A2[T] | None,
    surf_type: VtkEnum | None,
) -> CheartMeshBoundary[T] | None:
    if raw_bnd is None or surf_type is None:
        return None
    bnd_tags = np.unique(raw_bnd[:, -1])
    bnd = {int(tag): _create_bnd_surf(raw_bnd, int(tag), surf_type) for tag in bnd_tags}
    return CheartMeshBoundary(len(raw_bnd), bnd, surf_type)


def cheart_mesh_from_arrays[F: np.floating, I: np.integer](
    space: A2[F], top: A2[I], bnd: A2[I] | None = None, *, elem: VtkEnum | None = None
) -> Result[CheartMesh[F, I]]:
    el_dim = top.shape[1]
    b_dim = None if bnd is None else bnd.shape[1] - 2
    if elem is None:
        match guess_vtk_elem_from_dim(el_dim, b_dim):
            case Ok(elem): ...  # fmt: skip
            case Err(e):
                return Err(e)
    boundary_type = get_vtk_boundary_element(elem)
    topology = CheartMeshTopology(len(top), top, elem)
    boundary = _create_cheart_mesh_surf_from_raw(bnd, boundary_type)
    return Ok(CheartMesh(CheartMeshSpace(len(space), space), topology, boundary))


def import_cheart_mesh[F: np.floating, I: np.integer](
    name: Path | str,
    forced_type: VtkEnum | None = None,
    *,
    ftype: DType[F] = np.float64,
    itype: DType[I] = np.intp,
) -> Ok[CheartMesh[F, I]] | Err:
    prefix = fix_ch_sfx(str(name))
    raw_space = np.loadtxt(f"{prefix}X", dtype=ftype, skiprows=1)
    raw_top = np.loadtxt(f"{prefix}T", dtype=itype, skiprows=1) - 1
    raw_bnd = (
        np.loadtxt(f"{prefix}B", dtype=itype, skiprows=1) if Path(f"{prefix}B").is_file() else None
    )
    return cheart_mesh_from_arrays(raw_space, raw_top, raw_bnd, elem=forced_type).next()
