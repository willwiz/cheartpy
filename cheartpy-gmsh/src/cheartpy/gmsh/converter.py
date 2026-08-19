def convert_3d_to_msh_via_api(
    filename, nodes_coord, elements_conn, bnd_conns, group_idxs, bnd_names=None, group_names=None
):
    """Converts 3D Volumetric arrays (Tetrahedral or Hexahedral) to Gmsh MSH format.

    Parameters
    ----------
    - filename: Output file path (e.g., 'model.msh')
    - nodes_coord: 2D numpy float array of shape (N, 3) -> [X, Y, Z]
    - elements_conn: 2D numpy int array of shape (M, 4) for Teds or (M, 8) for Hexes
    - bnd_conns: List of 3 numpy 2D arrays containing surface face connectivities (Tri or Quad)
    - group_idxs: List of 3 numpy 1D arrays containing row indices of elements in elements_conn

    """
    gmsh.initialize()
    gmsh.model.add("numpy_3d_volume_mesh")

    num_nodes = len(nodes_coord)

    if bnd_names is None:
        bnd_names = ["Boundary_Surface_1", "Boundary_Surface_2", "Boundary_Surface_3"]
    if group_names is None:
        group_names = ["VolumeGroup_1", "VolumeGroup_2", "VolumeGroup_3"]

    # 1. DYNAMICALLY DETECT 3D ELEMENT TYPES
    nodes_per_elem = elements_conn.shape[1]

    if nodes_per_elem == 4:
        elem_type_id = 4  # 4-node Tetrahedron
        bnd_type_id = 2  # 3-node Triangle boundary faces
    elif nodes_per_elem == 8:
        elem_type_id = 5  # 8-node Hexahedron
        bnd_type_id = 3  # 4-node Quadrilateral boundary faces
    else:
        raise ValueError(
            f"Unsupported number of nodes per element: {nodes_per_elem}. Expected 4 (Tet) or 8 (Hex)."
        )

    # 2. ADD ALL NODES TO A GLOBAL DISCRETE VOLUME ENTITY
    # For 3D meshes, we store the global node pool inside a base volume entity (Dim=3, Tag=1)
    global_volume_tag = 1
    gmsh.model.addDiscreteEntity(dim=3, tag=global_volume_tag)

    node_tags = np.arange(1, num_nodes + 1)

    # Coordinates must be a flat 1D array: [x1, y1, z1, x2, y2, z2...]
    gmsh.model.mesh.addNodes(
        dim=3, tag=global_volume_tag, nodeTags=node_tags, coord=nodes_coord.flatten()
    )

    # 3. ADD BOUNDARY SURFACES (Dimension 2)
    bnd_element_offset = 1  # Track unique element IDs across the entire file

    for i in range(3):
        surface_tag = i + 1  # Tags: 1, 2, 3
        gmsh.model.addDiscreteEntity(dim=2, tag=surface_tag)

        bnd_data = bnd_conns[i]
        num_bnd_elems = len(bnd_data)
        bnd_tags = np.arange(bnd_element_offset, bnd_element_offset + num_bnd_elems)

        # Inject boundary faces into Dimension 2
        gmsh.model.mesh.addElements(
            dim=2,
            tag=surface_tag,
            elementTypes=[bnd_type_id],
            elementTags=[bnd_tags],
            nodeTags=[bnd_data.flatten()],
        )

        # Physical group for boundaries is now Dimension 2 (Surfaces)
        gmsh.model.addPhysicalGroup(dim=2, tags=[surface_tag], name=bnd_names[i])

        bnd_element_offset += num_bnd_elems

    # 4. ADD PHYSICAL VOLUMES / DOMAINS (Dimension 3)
    for i in range(3):
        # Tags: 2, 3, 4 (Since tag 1 was already used for the global node tracking entity)
        volume_tag = i + 2
        gmsh.model.addDiscreteEntity(dim=3, tag=volume_tag)

        target_indices = group_idxs[i]
        domain_data = elements_conn[target_indices]
        num_domain_elems = len(domain_data)
        domain_tags = np.arange(bnd_element_offset, bnd_element_offset + num_domain_elems)

        # Inject volumetric elements into Dimension 3
        gmsh.model.mesh.addElements(
            dim=3,
            tag=volume_tag,
            elementTypes=[elem_type_id],
            elementTags=[domain_tags],
            nodeTags=[domain_data.flatten()],
        )

        # Physical group for volumes is Dimension 3 (Volumes)
        gmsh.model.addPhysicalGroup(dim=3, tags=[volume_tag], name=group_names[i])

        bnd_element_offset += num_domain_elems

    # 5. EXPORT AND FINALIZE
    gmsh.write(filename)
    gmsh.finalize()
    print(f"Successfully converted 3D volumetric mesh to {filename}")
