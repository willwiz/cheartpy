from typing import TYPE_CHECKING, TypeGuard

import numpy as np
from pytools.logging import get_logger
from pytools.math import householder_orthogonal_basis, r_basis
from pytools.result import Err, Ok, Result, all_ok

from cheartpy.mesh import import_cheart_mesh
from cheartpy.mesh_tools.tools import MergedMesh, merge_meshes, normalize_by_row

from .normals import compute_surface_normal, create_mesh_from_surface

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from cheartpy.fe.aliases import EmbbededTopologyDef, TopologyDef
    from pytools.arrays import A2


def is_cutplane[T](top: TopologyDef[T]) -> TypeGuard[EmbbededTopologyDef[T]]:
    match top:
        case {"master": _, "bnd": _, "mesh": _}:
            return True
        case _:
            return False


def find_cutplane_master[T](*tops: EmbbededTopologyDef[T]) -> T:
    master_to_cutplane = {t["master"]: t for t in tops}
    match len(master_to_cutplane):
        case 1: ...  # fmt: skip
        case 0:
            msg = "No cutplane found"
            raise ValueError(msg)
        case _:
            msg = f"Multiple cutplanes found: {list(master_to_cutplane.keys())}"
            raise ValueError(msg)
    return master_to_cutplane.popitem()[0]


def compute_householder_basis[F: np.floating](normals: A2[F]) -> A2[F]:
    mean_normal = normals.mean(axis=0)
    print(f"{mean_normal=}")
    basis = householder_orthogonal_basis(mean_normal)
    return np.full((normals.shape[0], 9), basis.flatten())


def compute_zrc_basis[F: np.floating](space: A2[F], normals: A2[F]) -> Result[A2[F]]:
    centroid = space.mean(axis=0)
    mean_normal = normals.mean(axis=0)
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    z = np.full((normals.shape[0], 3), mean_normal)
    r = space - centroid
    r = r - np.einsum("ij,j,k->ik", r, mean_normal, mean_normal)
    match normalize_by_row(r):
        case Ok(r): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    c = np.cross(z, r)
    return Ok(np.concatenate((z, r, c), axis=1).astype(space.dtype))


def make_cutplane_topology[T](  # noqa: C901, PLR0911
    defn: Mapping[T, TopologyDef[T]],
    planes: Sequence[T],
    new_home: Path,
) -> Result[MergedMesh[np.float64, np.intp]]:
    get_logger(level="INFO")
    new_home.mkdir(parents=True, exist_ok=True)
    cutplanes = {
        k: t
        for k, t in zip(
            planes,
            [defn[name] for name in planes if name in defn],
            strict=True,
        )
        if is_cutplane(t)
    }
    master = find_cutplane_master(*cutplanes.values())
    match import_cheart_mesh(defn[master]["mesh"], ftype=np.float64, itype=np.intp):
        case Ok(master_mesh): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    if master_mesh.bnd is None:
        msg = f"Master mesh {master} has no boundary"
        return Err(ValueError(msg))
    match all_ok(
        {k: create_mesh_from_surface(master_mesh, pln["bnd"]) for k, pln in cutplanes.items()}
    ):
        case Ok(bnd_meshes): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    match all_ok(
        {k: compute_surface_normal(master_mesh, pln["bnd"]) for k, pln in cutplanes.items()}
    ):
        case Ok(bnd_normals): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    match all_ok({k: r_basis(normals, flatten=True) for k, normals in bnd_normals.items()}):
        case Ok(bnd_bases): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    match all_ok({k: compute_zrc_basis(bnd_meshes[k].space.v, bnd_normals[k]) for k in cutplanes}):
        case Ok(bnd_zrc_bases): ...  # fmt: skip
        case Err(e): return Err(e)  # fmt: skip
    ids = {k: pln["bnd"] * np.ones((bnd_meshes[k].space.n, 1)) for k, pln in cutplanes.items()}
    return merge_meshes(
        list(bnd_meshes.values()),
        {
            "Normal": list(bnd_normals.values()),
            "Basis": list(bnd_bases.values()),
            "IDs": list(ids.values()),
            "ZRC": list(bnd_zrc_bases.values()),
        },
    ).next()
