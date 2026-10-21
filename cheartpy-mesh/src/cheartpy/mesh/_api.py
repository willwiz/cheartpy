import bz2
import gzip
import lzma
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import (
    CheartEnum,
    get_boundary_element,
    guess_element_from_dim,
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
    from io import BufferedIOBase

    from pytools.arrays import A2, DType, ToInt


def _create_bnd_surf[T: np.integer](v: A2[T], tag: ToInt, kind: CheartEnum) -> CheartMeshPatch[T]:
    bnd = v[v[:, -1] == tag, :-1]
    elems = bnd[:, 0] - 1
    nodes = bnd[:, 1:] - 1
    return CheartMeshPatch(int(tag), elems, nodes, kind)


def _create_cheart_mesh_surf_from_raw[T: np.integer](
    raw_bnd: A2[T] | None,
    surf_type: CheartEnum | None,
) -> CheartMeshBoundary[T]:
    if raw_bnd is None or surf_type is None:
        return CheartMeshBoundary[T]({})
    bnd_tags = np.unique(raw_bnd[:, -1])
    bnd = {int(tag): _create_bnd_surf(raw_bnd, int(tag), surf_type) for tag in bnd_tags}
    return CheartMeshBoundary(bnd)


def cheart_mesh_from_arrays[F: np.floating, I: np.integer](
    space: A2[F], top: A2[I], bnd: A2[I] | None = None, *, elem: CheartEnum | None = None
) -> Result[CheartMesh[F, I]]:
    el_dim = top.shape[1]
    b_dim = None if bnd is None else bnd.shape[1] - 2
    if elem is None:
        match guess_element_from_dim(el_dim, b_dim, "Cheart"):
            case Ok(elem): ...  # fmt: skip
            case Err(e):
                return Err(e)
    boundary_type = get_boundary_element(elem)
    topology = CheartMeshTopology(top - 1, elem)
    boundary = _create_cheart_mesh_surf_from_raw(bnd, boundary_type)
    return Ok(CheartMesh(CheartMeshSpace(space), topology, boundary))


def _load_from_stream[F: np.floating, I: np.integer](
    name: BufferedIOBase,
    *,
    ftype: DType[F] = np.float64,
    itype: DType[I] = np.intp,
) -> Ok[CheartMesh[F, I]] | Err:
    with np.load(name) as mesh_archive:
        if "x" not in mesh_archive:
            return Err(ValueError("Field 'x' not found. Incorrect file format."))
        if "t" not in mesh_archive:
            return Err(ValueError("Field 't' not found. Incorrect file format."))
        raw_space = mesh_archive["x"].astype(ftype)
        raw_top = mesh_archive["t"].astype(itype)
        raw_bnd = mesh_archive["b"].astype(itype) if "b" in mesh_archive else None
    return cheart_mesh_from_arrays(raw_space, raw_top, raw_bnd).next()


def import_cheart_mesh_compressed[F: np.floating, I: np.integer](
    name: Path | str,
    *,
    ftype: DType[F] = np.float64,
    itype: DType[I] = np.intp,
) -> Ok[CheartMesh[F, I]] | Err:
    name = Path(name)
    if name.suffix == ".xz":
        with lzma.open(name, "rb") as f:
            return _load_from_stream(f, ftype=ftype, itype=itype)
    elif name.suffix == ".gz":
        with gzip.open(name, "rb") as f:
            return _load_from_stream(f, ftype=ftype, itype=itype)
    elif name.suffix == ".bz2":
        with bz2.open(name, "rb") as f:
            return _load_from_stream(f, ftype=ftype, itype=itype)
    else:
        with name.open("rb") as f:
            return _load_from_stream(f, ftype=ftype, itype=itype)


def import_cheart_mesh[F: np.floating, I: np.integer](
    name: Path | str,
    forced_type: CheartEnum | None = None,
    *,
    ftype: DType[F] = np.float64,
    itype: DType[I] = np.intp,
) -> Ok[CheartMesh[F, I]] | Err:
    if Path(name).name.endswith((".npz", ".npz.xz", ".npz.gz", ".npz.bz2")):
        return import_cheart_mesh_compressed(name, ftype=ftype, itype=itype)
    prefix = fix_ch_sfx(str(name))
    raw_space = np.loadtxt(f"{prefix}X", dtype=ftype, skiprows=1)
    raw_top = np.loadtxt(f"{prefix}T", dtype=itype, skiprows=1)
    raw_bnd = (
        np.loadtxt(f"{prefix}B", dtype=itype, skiprows=1) if Path(f"{prefix}B").is_file() else None
    )
    return cheart_mesh_from_arrays(raw_space, raw_top, raw_bnd, elem=forced_type).next()
