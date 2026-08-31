from typing import TYPE_CHECKING

from ._types import AbaqusEnum, CheartEnum, NodeOrder, VtkEnum

if TYPE_CHECKING:
    from collections.abc import Mapping

Abaqus2Vtk = {
    AbaqusEnum.LINE1: VtkEnum.LINE1,
    AbaqusEnum.LINE2: VtkEnum.LINE2,
    AbaqusEnum.TRIANGLE1: VtkEnum.TRIANGLE1,
    AbaqusEnum.TRIANGLE2: VtkEnum.TRIANGLE2,
    AbaqusEnum.QUADRILATERAL1: VtkEnum.QUADRILATERAL1,
    AbaqusEnum.QUADRILATERAL2: VtkEnum.QUADRILATERAL2,
    AbaqusEnum.TETRAHEDRON1: VtkEnum.TETRAHEDRON1,
    AbaqusEnum.TETRAHEDRON2: VtkEnum.TETRAHEDRON2,
    AbaqusEnum.HEXAHEDRON1: VtkEnum.HEXAHEDRON1,
    AbaqusEnum.HEXAHEDRON2: VtkEnum.HEXAHEDRON2,
    AbaqusEnum.S3R: VtkEnum.TRIANGLE1,
    AbaqusEnum.CPEG6: VtkEnum.TRIANGLE2,
}
Vtk2Abaqus = {v: k for k, v in Abaqus2Vtk.items()}

_Abaqus2Cheart = {
    AbaqusEnum.LINE1: CheartEnum.LINE1,
    AbaqusEnum.LINE2: CheartEnum.LINE2,
    AbaqusEnum.S3R: CheartEnum.TRIANGLE1,
    AbaqusEnum.TRIANGLE1: CheartEnum.TRIANGLE1,
    AbaqusEnum.TRIANGLE2: CheartEnum.TRIANGLE2,
    AbaqusEnum.CPEG6: CheartEnum.TRIANGLE2,
    AbaqusEnum.QUADRILATERAL1: CheartEnum.QUADRILATERAL1,
    AbaqusEnum.TETRAHEDRON1: CheartEnum.TETRAHEDRON1,
    AbaqusEnum.TETRAHEDRON2: CheartEnum.TETRAHEDRON2,
}

_Abaqus2CheartNodeOrder: Mapping[AbaqusEnum, tuple[int, ...]] = {
    AbaqusEnum.LINE1: (0, 1),
    AbaqusEnum.LINE2: (0, 1, 2),
    AbaqusEnum.S3R: (0, 1, 2),
    AbaqusEnum.TRIANGLE1: (0, 1, 2),
    AbaqusEnum.TRIANGLE2: (0, 1, 2, 3, 5, 4),
    AbaqusEnum.CPEG6: (0, 1, 2, 3, 5, 4),
    AbaqusEnum.QUADRILATERAL1: (0, 1, 2, 3),
    AbaqusEnum.TETRAHEDRON1: (0, 1, 2, 3),
    AbaqusEnum.TETRAHEDRON2: (0, 1, 2, 3, 4, 6, 5, 7, 8, 9),
}

_AbaqusBoundaryElement: Mapping[AbaqusEnum, AbaqusEnum] = {
    AbaqusEnum.S3R: AbaqusEnum.LINE1,
    AbaqusEnum.TRIANGLE1: AbaqusEnum.LINE1,
    AbaqusEnum.TRIANGLE2: AbaqusEnum.LINE2,
    AbaqusEnum.CPEG6: AbaqusEnum.LINE2,
    AbaqusEnum.QUADRILATERAL1: AbaqusEnum.LINE1,
    AbaqusEnum.TETRAHEDRON1: AbaqusEnum.TRIANGLE1,
    AbaqusEnum.TETRAHEDRON2: AbaqusEnum.CPEG6,
}


def convert_abaqus_to_vtk(elem: AbaqusEnum) -> VtkEnum | None:
    return Abaqus2Vtk.get(elem)


def get_abaqus_elem_from_vtk(elem: VtkEnum) -> AbaqusEnum | None:
    return Vtk2Abaqus.get(elem)


def convert_abaqus_to_cheart(elem: AbaqusEnum) -> CheartEnum | None:
    return _Abaqus2Cheart.get(elem)


def get_vtk_element_for_abaqus(body: AbaqusEnum) -> VtkEnum | None:
    return Abaqus2Vtk.get(body)


def get_cheart_element_for_abaqus(body: AbaqusEnum) -> CheartEnum | None:
    return _Abaqus2Cheart.get(body)


def get_abaqus_boundary_element(body: AbaqusEnum) -> AbaqusEnum | None:
    return _AbaqusBoundaryElement.get(body)


def get_cheart_order_for_abaqus(elem: AbaqusEnum) -> tuple[int, ...]:
    return _Abaqus2CheartNodeOrder[elem]


S3R = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0)}
CPEG6 = {0: (0, 0, 0), 1: (2, 0, 0), 2: (0, 2, 0), 3: (1, 0, 0), 4: (1, 1, 0), 5: (0, 1, 0)}
T3D2 = {0: (0, 0, 0), 1: (1, 0, 0)}
T3D3 = {0: (0, 0, 0), 1: (1, 0, 0), 2: (2, 0, 0)}
CPS3 = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0)}
CPS6 = {0: (0, 0, 0), 1: (2, 0, 0), 2: (0, 2, 0), 3: (1, 0, 0), 4: (1, 1, 0), 5: (0, 1, 0)}
CPS4 = {0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 1, 0), 3: (0, 1, 0)}
M3D9 = {
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
C3D4 = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)}
C3D10 = {
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
C3D8 = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (1, 1, 0),
    3: (0, 1, 0),
    4: (0, 0, 1),
    5: (1, 0, 1),
    6: (1, 1, 1),
    7: (0, 1, 1),
}
C3D27 = {
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
    20: (1, 1, 1),
    21: (1, 1, 0),
    22: (1, 1, 2),
    23: (1, 0, 1),
    24: (2, 1, 1),
    25: (1, 2, 1),
    26: (0, 1, 1),
}

ELEMENT_ORDER = {
    AbaqusEnum.S3R: S3R,
    AbaqusEnum.CPEG6: CPEG6,
    AbaqusEnum.LINE1: T3D2,
    AbaqusEnum.LINE2: T3D3,
    AbaqusEnum.TRIANGLE1: CPS3,
    AbaqusEnum.TRIANGLE2: CPS6,
    AbaqusEnum.QUADRILATERAL1: CPS4,
    AbaqusEnum.QUADRILATERAL2: M3D9,
    AbaqusEnum.TETRAHEDRON1: C3D4,
    AbaqusEnum.TETRAHEDRON2: C3D10,
    AbaqusEnum.HEXAHEDRON1: C3D8,
    AbaqusEnum.HEXAHEDRON2: C3D27,
}


def get_abaqus_element_order(elem: AbaqusEnum) -> NodeOrder:
    return ELEMENT_ORDER[elem]
