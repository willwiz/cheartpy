from pytools.result import Err, Ok, Result

from ._types import CheartEnum, VtkElemShape, VtkEnum

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
