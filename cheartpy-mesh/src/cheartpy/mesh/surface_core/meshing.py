import numpy as np
from cheartpy.mesh.struct import CheartMesh, CheartMeshSpace, CheartMeshTopology
from cheartpy.vtk.api import get_vtk_elem
from pytools.logging.trait import NULL_LOG, ILogger


def create_mesh_from_surface[F: np.floating, I: np.integer](
    body: CheartMesh[F, I],
    surf_id: int,
    *,
    log: ILogger = NULL_LOG,
) -> CheartMesh[F, I]:
    """Create a mesh from a surface defined in the boundary of a CheartMesh.

    Parameters
    ----------
    body : CheartMesh
        The mesh from which to extract the surface.
    surf_id : int
        The ID of the surface in the boundary of the mesh.
    log : ILogger, optional
        Logger for debug messages, by default NULL, i.e., don't so anything.

    Returns
    -------
    Mesh: CheartMesh
        A new mesh containing only the surface defined by the boundary.

    """
    body_elem = get_vtk_elem(body.top.TYPE)
    if body_elem.surf is None:
        log.error("Mesh is 1D, normal not defined")
        raise ValueError
    if body.bnd is None:
        log.error("Mesh has no boundary, cannot create surface mesh")
        raise ValueError
    surf = body.bnd.v.get(surf_id)
    if surf is None:
        log.error(f"Boundary {surf_id} not found in mesh")
        raise ValueError
    nodes = {k: v for v, k in enumerate(np.unique(surf.v))}
    space = CheartMeshSpace(len(nodes), body.space.v[list(nodes.keys())])
    top = CheartMeshTopology(
        surf.n,
        np.array([[nodes[i] for i in v] for v in surf.v], dtype=body.top.v.dtype),
        body_elem.surf,
    )
    return CheartMesh(space, top, None)
