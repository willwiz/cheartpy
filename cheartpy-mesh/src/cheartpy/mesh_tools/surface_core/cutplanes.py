from typing import TYPE_CHECKING, TypeGuard

import numpy as np
from cheartpy.elem_interfaces import get_vtk_boundary_element
from pytools.logging import get_logger
from pytools.math import householder_orthogonal_basis
from pytools.result import Err, Result

from cheartpy.mesh import import_cheart_mesh
from cheartpy.mesh_tools.surface_core import (
    compute_surface_normal,
    create_mesh_from_surface,
    normalize_by_row,
)
from cheartpy.mesh_tools.tools import MergedMesh, merge_meshes

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
    basis = householder_orthogonal_basis(mean_normal)
    return np.full((normals.shape[0], 9), basis.flatten())


def compute_zrc_basis[F: np.floating](space: A2[F], normals: A2[F]) -> A2[F]:
    centroid = space.mean(axis=0)
    mean_normal = normals.mean(axis=0)
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    z = np.full((normals.shape[0], 3), mean_normal)
    r = space - centroid
    r = r - np.einsum("ij,j,k->ik", r, mean_normal, mean_normal)
    r = normalize_by_row(r)
    c = np.cross(z, r)
    return np.concatenate((z, r, c), axis=1).astype(space.dtype)


def make_cutplane_topology[T](
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
    master_mesh = import_cheart_mesh(defn[master]["mesh"], ftype=np.float64, itype=np.intp).unwrap()
    master_mesh.save(new_home / "Lin")
    if master_mesh.bnd is None:
        msg = f"Master mesh {master} has no boundary"
        return Err(ValueError(msg))
    bnd_type = get_vtk_boundary_element(master_mesh.bnd.TYPE)
    if bnd_type is None:
        msg = f"Unsupported boundary type {master_mesh.bnd.TYPE}"
        return Err(ValueError(msg))
    bnd_meshes = {
        k: create_mesh_from_surface(master_mesh, pln["bnd"]).unwrap()
        for k, pln in cutplanes.items()
    }
    bnd_normals = {
        k: compute_surface_normal(master_mesh, pln["bnd"]).unwrap() for k, pln in cutplanes.items()
    }
    bnd_bases = {k: compute_householder_basis(normals) for k, normals in bnd_normals.items()}
    bnd_zrc_bases = {k: compute_zrc_basis(bnd_meshes[k].space.v, bnd_normals[k]) for k in cutplanes}
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
