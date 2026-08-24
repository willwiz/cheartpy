from pprint import pprint
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.mesh import CheartMesh, CheartMeshBoundary, CheartMeshPatch
from pytools.result import Err, Ok, Result

from ._types import BoundaryAssociation, MultiDomainMesh

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import A1


def classify_boundary_to_region[F: np.floating, I: np.integer](
    patch: CheartMeshPatch[I], elems: A1[np.integer]
) -> BoundaryAssociation:
    print("checking for patch", patch.tag)
    assoc = np.isin(patch.k, elems, assume_unique=True)
    if np.all(assoc):
        print("full association")
        return BoundaryAssociation.FULL
    if np.any(assoc):
        print("partial association")
        return BoundaryAssociation.PARTIAL
    print("no association")
    return BoundaryAssociation.NONE


def find_subdomain_boundary[F: np.floating, I: np.integer](
    bnd: CheartMeshBoundary[I], region_elems: A1[np.integer], k: int
) -> CheartMeshBoundary[I]:
    print(f"Finding subdomain boundary for region {k}")
    patches = {
        k: v
        for k, v in bnd.v.items()
        if classify_boundary_to_region(v, region_elems) is BoundaryAssociation.FULL
    }
    return CheartMeshBoundary(len(patches), patches, bnd.TYPE)


def split_subdomain[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], regions: Mapping[int, A1[np.integer]] | None = None
) -> Result[MultiDomainMesh[F, I]]:
    """Split a CheartMesh into subdomains based on region labels.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The input CheartMesh to be split.
    regions : np.ndarray
        An array of region labels corresponding to each element in the mesh.

    Returns
    -------
    MultiDomainMesh[F, I]
        A MultiDomainMesh containing the volume mesh and subdomain information.

    """
    if not regions:
        return Ok(
            MultiDomainMesh(mesh, {1: np.arange(mesh.top.n, dtype=mesh.top.v.dtype)}, {1: mesh.bnd})
        )
    if not mesh.bnd:
        msg = "Mesh has no boundary information, cannot split into subdomains."
        return Err(ValueError(msg))
    region_bnds = {k: find_subdomain_boundary(mesh.bnd, v, k) for k, v in regions.items()}
    pprint({k: {t: (v.tag, v.TYPE.name) for t, v in b.v.items()} for k, b in region_bnds.items()})
    return Ok(
        MultiDomainMesh(
            mesh, {k: v.astype(mesh.top.v.dtype) for k, v in regions.items()}, region_bnds
        )
    )
