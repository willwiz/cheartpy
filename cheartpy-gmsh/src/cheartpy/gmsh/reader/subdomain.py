from pprint import pprint
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.gmsh.tools import (
    GmshBoundaryType,
    build_element_searchmap,
    search_element_association,
)
from pytools.result import Err, Ok, Result

from ._types import BoundaryAssociation, GmshBndClass, MultiDomainMesh

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cheartpy.mesh import CheartMesh, CheartMeshBoundary, CheartMeshPatch
    from pytools.arrays import A1


def classify_boundary_to_region[F: np.floating, I: np.integer](
    patch: CheartMeshPatch[I],
    search_map: Mapping[int, set[int]],
) -> tuple[BoundaryAssociation, GmshBoundaryType]:
    """Determine the relation between a boundary patch and a region of elements.

    Parameters
    ----------
    patch : CheartMeshPatch[I]
        The boundary patch to classify.
    search_map : Mapping[int, set[int]]
        A mapping from node indices to sets of element indices that contain those nodes.

    Returns
    -------
    BoundaryAssociation
        An enumeration indicating whether the boundary patch is:
        -   fully associated all patches are in elements of the region,
        -   partially associated some patches are in elements of the region, or
        -   not associated no patches are in elements of the region.

    """
    print("    Checking for patch", patch.tag, end=": ")
    search_results = [search_element_association(search_map, n) for n in patch.v]
    found = np.array([r is not GmshBoundaryType.NONE for r in search_results])
    kind = set(search_results)
    if len(kind) > 1:
        msg = "Boundary patch has mixed association with the region: " + " + ".join(
            str(k) for k in kind
        )
        raise ValueError(msg)
    if np.all(found):
        assoc = BoundaryAssociation.FULL
    elif np.any(found):
        assoc = BoundaryAssociation.PARTIAL
    else:
        assoc = BoundaryAssociation.NONE
    print(f"{assoc.name!s}")
    return assoc, kind.pop()


def find_subdomain_boundary[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], bnd: CheartMeshBoundary[I], region_elems: A1[np.integer], k: int
) -> Mapping[int, GmshBndClass[I]]:
    """Find all of the boundary patches that are associated with a given region of elements."""
    print(f">>> Finding subdomain boundary for region {k}")
    search_map = build_element_searchmap(region_elems, mesh.top.v[region_elems])
    patch_assoc = {k: (classify_boundary_to_region(v, search_map)) for k, v in bnd.v.items()}
    patches: Mapping[int, GmshBndClass[I]] = {
        k: {"patch": v, "kind": patch_assoc[k][1]}
        for k, (v) in bnd.v.items()
        if patch_assoc[k][0] is BoundaryAssociation.FULL
    }
    print(patches.keys())
    return patches


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
        regions = {1: np.arange(len(mesh.top.v), dtype=mesh.top.v.dtype)}
    if not mesh.bnd:
        msg = "Mesh has no boundary information, cannot split into subdomains."
        return Err(ValueError(msg))
    region_bnds = {k: find_subdomain_boundary(mesh, mesh.bnd, v, k) for k, v in regions.items()}
    pprint(
        {
            k: {t: (v["patch"].tag, v["patch"].TYPE.name) for t, v in b.items()}
            for k, b in region_bnds.items()
        }
    )
    return Ok(
        MultiDomainMesh(
            mesh, {k: v.astype(mesh.top.v.dtype) for k, v in regions.items()}, region_bnds
        )
    )
