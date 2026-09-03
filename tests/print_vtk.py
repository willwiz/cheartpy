from pprint import pprint

from cheartpy.elem_interfaces import (
    CheartEnum,
    VtkEnum,
    get_cheart_elem_nodes,
    get_node_permutation,
    get_vtk_elem_nodes,
)


def compute_node_mapping(vtk_elem: VtkEnum, cheart_elem: CheartEnum) -> dict[int, int]:
    """Compute the mapping of node indices from VTK to Cheart for a given element type.

    Args:
        vtk_elem (VtkEnum): The VTK element type.
        cheart_elem (CheartEnum): The Cheart element type.

    Returns:
        list[int]: A list of indices representing the mapping from VTK node order to Cheart node order.

    """
    vtk_nodes = get_vtk_elem_nodes(vtk_elem)
    cheart_nodes = get_cheart_elem_nodes(cheart_elem)

    if len(vtk_nodes) != len(cheart_nodes):
        msg = "VTK and Cheart elements must have the same number of nodes."
        raise ValueError(msg)
    cheart_map = {v: k for k, v in cheart_nodes.items()}
    return {k: cheart_map[v] for k, v in vtk_nodes.items()}


def main() -> None:
    print("Testing get_cheart_elem_nodes and get_vtk_elem_nodes...")

    tests = {
        "line1": (VtkEnum.LINE1, CheartEnum.LINE1),
        "line2": (VtkEnum.LINE2, CheartEnum.LINE2),
        "triangle1": (VtkEnum.TRIANGLE1, CheartEnum.TRIANGLE1),
        "triangle2": (VtkEnum.TRIANGLE2, CheartEnum.TRIANGLE2),
        "quadrilateral1": (VtkEnum.QUADRILATERAL1, CheartEnum.QUADRILATERAL1),
        "quadrilateral2": (VtkEnum.QUADRILATERAL2, CheartEnum.QUADRILATERAL2),
        "tetrahedron1": (VtkEnum.TETRAHEDRON1, CheartEnum.TETRAHEDRON1),
        "tetrahedron2": (VtkEnum.TETRAHEDRON2, CheartEnum.TETRAHEDRON2),
        "hexahedron1": (VtkEnum.HEXAHEDRON1, CheartEnum.HEXAHEDRON1),
        "hexahedron2": (VtkEnum.HEXAHEDRON2, CheartEnum.HEXAHEDRON2),
    }
    for name, (v, c) in tests.items():
        print(f"{name!s}:")
        dct = compute_node_mapping(v, c)
        perm = get_node_permutation(v, "Cheart")
        print(f"Permutation from VTK to Cheart: {perm}")
        pprint({v: k for k, v in dct.items()}, sort_dicts=True)
        pprint(dct)
        print(get_node_permutation(c, "Vtk"))


if __name__ == "__main__":
    main()
