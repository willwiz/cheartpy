from pytools.result import Err, Ok, Result

from ._types import CheartEnum, NodeOrder, VtkElemShape, VtkEnum

_Vtk2Cheart = {
    VtkEnum.LINE1: CheartEnum.LINE1,
    VtkEnum.TRIANGLE1: CheartEnum.TRIANGLE1,
    VtkEnum.QUADRILATERAL1: CheartEnum.QUADRILATERAL1,
    VtkEnum.TETRAHEDRON1: CheartEnum.TETRAHEDRON1,
    VtkEnum.HEXAHEDRON1: CheartEnum.HEXAHEDRON1,
    VtkEnum.LINE2: CheartEnum.LINE2,
    VtkEnum.TRIANGLE2: CheartEnum.TRIANGLE2,
    VtkEnum.QUADRILATERAL2: CheartEnum.QUADRILATERAL2,
    VtkEnum.TETRAHEDRON2: CheartEnum.TETRAHEDRON2,
    VtkEnum.HEXAHEDRON2: CheartEnum.HEXAHEDRON2,
}


# fmt: off
_Vtk2CheartNodeOrder = {
    VtkEnum.LINE1: (0, 1),
    VtkEnum.TRIANGLE1: (0, 1, 2),
    VtkEnum.QUADRILATERAL1: (0, 1, 3, 2),
    VtkEnum.TETRAHEDRON1: (0, 1, 2, 3),
    VtkEnum.HEXAHEDRON1: (0, 1, 5, 4, 2, 3, 7, 6),
    VtkEnum.LINE2: (0, 1, 2),
    VtkEnum.TRIANGLE2: (0, 1, 2, 3, 5, 4),
    VtkEnum.QUADRILATERAL2: (0, 1, 3, 2, 4, 7, 8, 5, 6),
    VtkEnum.TETRAHEDRON2: (0, 1, 2, 3, 4, 6, 5, 7, 8, 9),
    VtkEnum.HEXAHEDRON2: (
        0,  1,  5,  4,  2,  3,  7, 6,  8,  15,
        22, 13, 12, 21, 26, 19, 9, 11, 25, 23,
        16, 18, 10, 24, 14, 20, 17,
    ),
}
# fmt: on

_VtkBoundaryElement: dict[VtkEnum, VtkEnum] = {
    VtkEnum.TRIANGLE1: VtkEnum.LINE1,
    VtkEnum.QUADRILATERAL1: VtkEnum.LINE1,
    VtkEnum.TETRAHEDRON1: VtkEnum.TRIANGLE1,
    VtkEnum.HEXAHEDRON1: VtkEnum.QUADRILATERAL1,
    VtkEnum.TRIANGLE2: VtkEnum.LINE2,
    VtkEnum.QUADRILATERAL2: VtkEnum.LINE2,
    VtkEnum.TETRAHEDRON2: VtkEnum.TRIANGLE2,
    VtkEnum.HEXAHEDRON2: VtkEnum.QUADRILATERAL2,
}


def convert_vtk_to_cheart(elem: VtkEnum) -> CheartEnum | None:
    return _Vtk2Cheart.get(elem)


def get_cheart_order_for_vtk(elem: VtkEnum) -> tuple[int, ...]:
    return _Vtk2CheartNodeOrder[elem]


def get_vtk_boundary_element(elem: VtkEnum) -> VtkEnum | None:
    return _VtkBoundaryElement.get(elem)


def guess_vtk_elem_from_dim(edim: int, bdim: int | None) -> Result[VtkEnum]:
    match edim, bdim:
        case 3, 2 | None:
            elem = VtkEnum.TRIANGLE1
        case 6, 3 | None:
            elem = VtkEnum.TRIANGLE2
        case 4, 2:
            elem = VtkEnum.QUADRILATERAL1
        case 9, 3 | None:
            elem = VtkEnum.QUADRILATERAL2
        case 4, 3:
            elem = VtkEnum.TETRAHEDRON1
        case 10, 6 | None:
            elem = VtkEnum.TETRAHEDRON2
        case 8, 4 | None:
            elem = VtkEnum.HEXAHEDRON1
        case 27, 9 | None:
            elem = VtkEnum.HEXAHEDRON2
        case 4, None:
            msg = (
                "Cannot detect between Bilinear quadrilateral and Trilinear tetrahedron,"
                "need boundary dim"
            )
            return Err(ValueError(msg))
        case _:
            msg = f"Unsupported element dimensions: edim={edim}, bdim={bdim}"
            return Err(ValueError(msg))
    return Ok(elem)


_VtkEnumCategory: dict[tuple[VtkElemShape, int], VtkEnum] = {
    ("Line", 1): VtkEnum.LINE1,
    ("Triangle", 1): VtkEnum.TRIANGLE1,
    ("Quadrilateral", 1): VtkEnum.QUADRILATERAL1,
    ("Tetrahedron", 1): VtkEnum.TETRAHEDRON1,
    ("Hexahedron", 1): VtkEnum.HEXAHEDRON1,
    ("Line", 2): VtkEnum.LINE2,
    ("Triangle", 2): VtkEnum.TRIANGLE2,
    ("Quadrilateral", 2): VtkEnum.QUADRILATERAL2,
    ("Tetrahedron", 2): VtkEnum.TETRAHEDRON2,
    ("Hexahedron", 2): VtkEnum.HEXAHEDRON2,
}


def get_vtkelem_with_polyorder(elem: VtkEnum, order: int) -> VtkEnum | None:
    return _VtkEnumCategory.get((elem.value.shape, order))


LINE1 = {
    0: (0, 0, 0),
    1: (1, 0, 0),
}
LINE2 = {
    0: (0, 0, 0),
    1: (2, 0, 0),
    2: (1, 0, 0),
}
TRI1 = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (0, 1, 0),
}
TRI2 = {
    0: (0, 0, 0),
    1: (2, 0, 0),
    2: (0, 2, 0),
    3: (1, 0, 0),
    4: (1, 1, 0),
    5: (0, 1, 0),
}
QUAD1 = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (1, 1, 0),
    3: (0, 1, 0),
}
QUAD2 = {
    0: (0, 0, 0),
    1: (2, 0, 0),
    2: (2, 2, 0),
    3: (0, 2, 0),
    4: (1, 0, 0),
    5: (2, 1, 0),
    6: (1, 2, 0),
    7: (0, 1, 0),
    8: (1, 1, 0),
}
TET1 = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (0, 1, 0),
    3: (0, 0, 1),
}
TET2 = {
    0: (0, 0, 0),
    1: (2, 0, 0),
    2: (0, 2, 0),
    3: (0, 0, 2),
    4: (1, 0, 0),
    5: (1, 1, 0),
    6: (0, 1, 0),
    7: (0, 0, 1),
    8: (1, 0, 1),
    9: (0, 1, 1),
}
# Do here .....
HEX1 = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (1, 1, 0),
    3: (0, 1, 0),
    4: (0, 0, 1),
    5: (1, 0, 1),
    6: (1, 1, 1),
    7: (0, 1, 1),
}
HEX2 = {
    0: (0, 0, 0),
    1: (2, 0, 0),
    2: (2, 2, 0),
    3: (0, 2, 0),
    4: (0, 0, 2),
    5: (2, 0, 2),
    6: (2, 2, 2),
    7: (0, 2, 2),
    8: (1, 0, 0),
    9: (2, 1, 0),
    10: (1, 2, 0),
    11: (0, 1, 0),
    12: (1, 0, 2),
    13: (2, 1, 2),
    14: (1, 2, 2),
    15: (0, 1, 2),
    16: (0, 0, 1),
    17: (2, 0, 1),
    18: (2, 2, 1),
    19: (0, 2, 1),
    20: (0, 1, 1),
    21: (2, 1, 1),
    22: (1, 0, 1),
    23: (1, 2, 1),
    24: (1, 1, 0),
    25: (1, 1, 2),
    26: (1, 1, 1),
}


VTK_ELEMENT_NODES: dict[VtkEnum, dict[int, tuple[int, int, int]]] = {
    VtkEnum.LINE1: LINE1,
    VtkEnum.LINE2: LINE2,
    VtkEnum.TRIANGLE1: TRI1,
    VtkEnum.TRIANGLE2: TRI2,
    VtkEnum.QUADRILATERAL1: QUAD1,
    VtkEnum.QUADRILATERAL2: QUAD2,
    VtkEnum.TETRAHEDRON1: TET1,
    VtkEnum.TETRAHEDRON2: TET2,
    VtkEnum.HEXAHEDRON1: HEX1,
    VtkEnum.HEXAHEDRON2: HEX2,
}


def get_vtk_elem_nodes(elem: VtkEnum) -> NodeOrder:
    return VTK_ELEMENT_NODES[elem]
