from typing import TYPE_CHECKING, NamedTuple, Protocol, Unpack

import numpy as np
from cheartpy.elem_interfaces import get_cheart_elem_from_vtk
from cheartpy.elem_interfaces._gmsh import Vtk2Gmsh
from cheartpy.elem_interfaces._remapping import Cheart2VtkNodeOrder
from typing_extensions import TypedDict

import gmsh

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from cheartpy.mesh import CheartMesh, CheartMeshPatch
    from pytools.arrays import A1, A2


class IndexGenerator(Protocol):
    def __call__(self) -> int: ...


class Counter:
    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> int:
        self.count = self.count + 1
        return self.count


class GmshTopInfo(NamedTuple):
    elem_tags: A1[np.integer]
    connectivity: A2[np.integer]
    vol_type_id: int
    dim: int


new_entity: IndexGenerator = Counter()  # Unique entity tag generator
new_surface: IndexGenerator = Counter()  # Unique surface tag generator


def add_cheart_top_to_gmsh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], current_elem: int = 1
) -> tuple[int, GmshTopInfo]:
    num_nodes, dim = mesh.space.v.shape
    n_elem, _ = mesh.top.v.shape

    vol_elem = Vtk2Gmsh[mesh.top.TYPE]
    cheart_elem = get_cheart_elem_from_vtk(mesh.top.TYPE)
    if cheart_elem is None:
        msg = f"Unsupported element type: {mesh.top.TYPE}"
        raise ValueError(msg)
    element_reorder = Cheart2VtkNodeOrder[cheart_elem]
    connectivity = np.ascontiguousarray(mesh.top.v[:, element_reorder] + 1)
    # 1. DYNAMICALLY DETECT 3D ELEMENT TYPES
    vol_type_id = vol_elem.value

    # 2. ADD ALL NODES TO A GLOBAL DISCRETE VOLUME ENTITY
    # For 3D meshes, we store the global node pool inside a base volume entity (Dim=3, Tag=1)
    global_volume_tag = new_entity()
    gmsh.model.add_discrete_entity(dim=dim, tag=global_volume_tag)
    node_tags = np.arange(1, num_nodes + 1)

    # Coordinates must be a flat 1D array: [x1, y1, z1, x2, y2, z2...]
    gmsh.model.mesh.add_nodes(
        dim=dim, tag=global_volume_tag, nodeTags=node_tags, coord=mesh.space.v.flatten()
    )
    current_elem = 1
    elem_tags = np.arange(current_elem, current_elem + n_elem)
    current_elem = current_elem + n_elem
    gmsh.model.mesh.add_elements(
        dim=dim,
        tag=global_volume_tag,
        elementTypes=[vol_type_id],
        elementTags=[elem_tags],
        nodeTags=[connectivity.flatten()],
    )
    gmsh.model.add_physical_group(dim=dim, tags=[global_volume_tag], name="Volume1")
    return current_elem, GmshTopInfo(elem_tags, connectivity, vol_type_id, dim)


def add_cheart_boundary_to_gmsh[I: np.integer](
    bnd: CheartMeshPatch[I], dim: int, current_elem: int
) -> int:
    surface_tag = new_entity()  # Tags: 1, 2, 3
    bnd_elem = Vtk2Gmsh[bnd.TYPE]
    cheart_elem = get_cheart_elem_from_vtk(bnd.TYPE)
    if cheart_elem is None:
        msg = f"Unsupported boundary element type: {bnd.TYPE}"
        raise ValueError(msg)
    bnd_reorder = Cheart2VtkNodeOrder[cheart_elem]
    bnd_type_id = bnd_elem.value
    gmsh.model.add_discrete_entity(dim=dim - 1, tag=surface_tag)

    bnd_data = bnd.v[:, bnd_reorder] + 1
    num_bnd_elems = len(bnd_data)
    bnd_tags = np.arange(current_elem, current_elem + num_bnd_elems)

    # Inject boundary faces into Dimension 2
    gmsh.model.mesh.add_elements(
        dim=dim - 1,
        tag=surface_tag,
        elementTypes=[bnd_type_id],
        elementTags=[bnd_tags],
        nodeTags=[bnd_data.flatten()],
    )

    # Physical group for boundaries is now Dimension 2 (Surfaces)
    gmsh.model.add_physical_group(dim=dim - 1, tags=[surface_tag], name=f"Surface{bnd.tag}")
    return current_elem + num_bnd_elems


def add_physical_group_by_elset[F: np.floating, I: np.integer](
    top: GmshTopInfo, name: str, elset: A1[I]
) -> None:
    volume_tag = new_entity()
    gmsh.model.add_discrete_entity(dim=top.dim, tag=volume_tag)
    domain_data = top.connectivity[elset]
    # Inject volumetric elements into Dimension 3
    gmsh.model.mesh.add_elements(
        dim=top.dim,
        tag=volume_tag,
        elementTypes=[top.vol_type_id],
        elementTags=[top.elem_tags[elset]],
        nodeTags=[domain_data.flatten()],
    )
    # Physical group for volumes is Dimension 3 (Volumes)
    gmsh.model.add_physical_group(dim=top.dim, tags=[volume_tag], name=name)


class MeshConverterKwargs(TypedDict, total=False):
    optimize: bool
    angle_deg: float


def optimize_mesh(**kwargs: Unpack[MeshConverterKwargs]) -> None:
    """Optimize the mesh using Gmsh's built-in optimization algorithms.

    Parameters
    ----------
    angle_deg : float, default=40.0
        The angle threshold in degrees for classifying surfaces. Surfaces with angles below this
        threshold will be considered for optimization.

    """
    angle_deg = kwargs.get("angle_deg", 40.0)
    gmsh.model.mesh.classify_surfaces(
        angle_deg * np.pi / 180.0, boundary=True, forReparametrization=True
    )
    # gmsh.model.mesh.create_geometry()
    gmsh.model.mesh.optimize("UntangleMeshGeometry")
    gmsh.model.mesh.optimize("Gmsh")
    gmsh.model.mesh.optimize("Netgen")
    gmsh.model.mesh.optimize("Gmsh")
    gmsh.model.mesh.optimize("Relocate3D")
    gmsh.model.mesh.optimize("Relocate3D")


def convert_3d_to_msh_via_api[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    regions: Mapping[str, A1[np.integer]] | None = None,
    filename: Path | None = None,
    **kwargs: Unpack[MeshConverterKwargs],
) -> None:
    """Convert 3D Volumetric arrays (Tetrahedral or Hexahedral) to Gmsh MSH format.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The 3D volumetric mesh to convert.
    filename : Path
        The output filename for the Gmsh MSH file.
    regions : Mapping[str, A1[I]] | None
        Optional mapping of region names to element indices for defining physical groups.
    optimize : bool, default=False
        Whether to optimize the mesh using Gmsh's built-in optimization algorithms.
    angle_deg : float, default=40.0
        The angle threshold in degrees for classifying surfaces during optimization.

    """
    gmsh.initialize()
    gmsh.model.add("3D_Volumetric_Mesh")

    current_elem, top = add_cheart_top_to_gmsh(mesh, current_elem=1)
    # 3. ADD BOUNDARY SURFACES (Dimension 2)
    if mesh.bnd is not None:
        for v in mesh.bnd.v.values():
            current_elem = add_cheart_boundary_to_gmsh(v, top.dim, current_elem)

    # 4. ADD PHYSICAL VOLUMES / DOMAINS (Dimension 3)
    if regions is not None:
        for k, v in regions.items():
            add_physical_group_by_elset(top, k, v)

    if kwargs.get("optimize", False):
        optimize_mesh(**kwargs)

    # 5. EXPORT AND FINALIZE
    if filename:
        gmsh.write(str(filename))
    else:
        gmsh.fltk.run()
    gmsh.finalize()
    print(f"Successfully converted 3D volumetric mesh to {filename}")
