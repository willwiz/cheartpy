from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces._gmsh import Vtk2Gmsh

import gmsh

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from cheartpy.mesh import CheartMesh
    from pytools.arrays import A1


class Counter:
    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> int:
        self.count = self.count + 1
        return self.count


def convert_3d_to_msh_via_api[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], filename: Path, regions: Mapping[str, A1[I]] | None = None
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

    """
    new_entity = Counter()  # Unique entity tag generator
    new_surface = Counter()  # Unique surface tag generator
    gmsh.initialize()
    gmsh.model.add("3D_Volumetric_Mesh")

    num_nodes, dim = mesh.space.v.shape
    n_elem, _ = mesh.top.v.shape

    vol_elem = Vtk2Gmsh[mesh.top.TYPE]

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
        nodeTags=[mesh.top.v.flatten() + 1],
    )
    gmsh.model.add_physical_group(dim=dim, tags=[global_volume_tag], name="Volume1")

    # 3. ADD BOUNDARY SURFACES (Dimension 2)
    if mesh.bnd is not None:
        for k, v in mesh.bnd.v.items():
            surface_tag = new_surface()  # Tags: 1, 2, 3
            bnd_elem = Vtk2Gmsh[mesh.bnd.TYPE]
            bnd_type_id = bnd_elem.value
            gmsh.model.add_discrete_entity(dim=dim - 1, tag=surface_tag)

            bnd_data = v.v + 1
            num_bnd_elems = len(bnd_data)
            bnd_tags = np.arange(current_elem, current_elem + num_bnd_elems)
            current_elem = current_elem + num_bnd_elems

            # Inject boundary faces into Dimension 2
            gmsh.model.mesh.add_elements(
                dim=dim - 1,
                tag=surface_tag,
                elementTypes=[bnd_type_id],
                elementTags=[bnd_tags],
                nodeTags=[bnd_data.flatten()],
            )

            # Physical group for boundaries is now Dimension 2 (Surfaces)
            gmsh.model.add_physical_group(dim=dim - 1, tags=[surface_tag], name=f"Surface{k}")

    # 4. ADD PHYSICAL VOLUMES / DOMAINS (Dimension 3)
    if regions is not None:
        for k, v in regions.items():
            # Tags: 2, 3, 4 (Since tag 1 was already used for the global node tracking entity)
            volume_tag = new_entity()
            gmsh.model.add_discrete_entity(dim=dim, tag=volume_tag)
            domain_data = mesh.top.v[v] + 1
            # Inject volumetric elements into Dimension 3
            gmsh.model.mesh.add_elements(
                dim=dim,
                tag=volume_tag,
                elementTypes=[vol_type_id],
                elementTags=[elem_tags[v]],
                nodeTags=[domain_data.flatten()],
            )
            # Physical group for volumes is Dimension 3 (Volumes)
            gmsh.model.add_physical_group(dim=dim, tags=[volume_tag], name=k)

    # 5. EXPORT AND FINALIZE
    gmsh.write(str(filename))
    gmsh.finalize()
    print(f"Successfully converted 3D volumetric mesh to {filename}")
