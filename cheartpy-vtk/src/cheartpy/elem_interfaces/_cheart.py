from ._types import CheartEnum, VtkEnum

Vtk2Cheart = {
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
Cheart2Vtk = {v: k for k, v in Vtk2Cheart.items()}


def get_cheart_elem_from_vtk(elem: VtkEnum) -> CheartEnum | None:
    return Vtk2Cheart.get(elem)


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
    4: (0, 1, 0),
    5: (1, 1, 0),
}
QUAD1 = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (0, 1, 0),
    3: (1, 1, 0),
}
QUAD2 = {
    0: (0, 0, 0),
    1: (2, 0, 0),
    2: (0, 2, 0),
    3: (2, 2, 0),
    4: (1, 0, 0),
    5: (0, 1, 0),
    6: (1, 1, 0),
    7: (2, 1, 0),
    8: (1, 2, 0),
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
    5: (0, 1, 0),
    6: (1, 1, 0),
    7: (0, 0, 1),
    8: (1, 0, 1),
    9: (0, 1, 1),
}
HEX1 = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (0, 1, 0),
    3: (1, 1, 0),
    4: (0, 0, 1),
    5: (1, 0, 1),
    6: (0, 1, 1),
    7: (1, 1, 1),
}
HEX2 = {
    0: (0, 0, 0),
    1: (2, 0, 0),
    2: (0, 2, 0),
    3: (2, 2, 0),
    4: (0, 0, 2),
    5: (2, 0, 2),
    6: (0, 2, 2),
    7: (2, 2, 2),
    8: (1, 0, 0),
    9: (0, 1, 0),
    10: (1, 1, 0),
    11: (2, 1, 0),
    12: (1, 2, 0),
    13: (0, 0, 1),
    14: (1, 0, 1),
    15: (2, 0, 1),
    16: (0, 1, 1),
    17: (1, 1, 1),
    18: (2, 1, 1),
    19: (0, 2, 1),
    20: (1, 2, 1),
    21: (2, 2, 1),
    22: (1, 0, 2),
    23: (0, 1, 2),
    24: (1, 1, 2),
    25: (2, 1, 2),
    26: (1, 2, 2),
}
