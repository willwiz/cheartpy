from .types import CheartEnum, VtkEnum

Vtk2Cheart = {
    VtkEnum.VtkLinearLine: CheartEnum.LINE1,
    VtkEnum.VtkLinearTriangle: CheartEnum.TRIANGLE1,
    VtkEnum.VtkLinearQuadrilateral: CheartEnum.QUADRILATERAL1,
    VtkEnum.VtkLinearTetrahedron: CheartEnum.TETRAHEDRON1,
    VtkEnum.VtkLinearHexahedron: CheartEnum.HEXAHEDRON1,
    VtkEnum.VtkQuadraticLine: CheartEnum.LINE2,
    VtkEnum.VtkQuadraticTriangle: CheartEnum.TRIANGLE2,
    VtkEnum.VtkQuadraticQuadrilateral: CheartEnum.QUADRILATERAL2,
    VtkEnum.VtkQuadraticTetrahedron: CheartEnum.TETRAHEDRON2,
    VtkEnum.VtkQuadraticHexahedron: CheartEnum.HEXAHEDRON2,
}


# fmt: off
Vtk2CheartNodeOrder = {
    VtkEnum.VtkLinearLine: (0, 1),
    VtkEnum.VtkLinearTriangle: (0, 1, 2),
    VtkEnum.VtkLinearQuadrilateral: (0, 1, 3, 2),
    VtkEnum.VtkLinearTetrahedron: (0, 1, 2, 3),
    VtkEnum.VtkLinearHexahedron: (0, 1, 5, 4, 2, 3, 7, 6),
    VtkEnum.VtkQuadraticLine: (0, 1, 2),
    VtkEnum.VtkQuadraticTriangle: (0, 1, 2, 3, 5, 4),
    VtkEnum.VtkQuadraticQuadrilateral: (0, 1, 3, 2, 4, 7, 8, 5, 6),
    VtkEnum.VtkQuadraticTetrahedron: (0, 1, 2, 3, 4, 6, 5, 7, 8, 9),
    VtkEnum.VtkQuadraticHexahedron: (
        0,  1,  5,  4,  2,  3,  7, 6,  8,  15,
        22, 13, 12, 21, 26, 19, 9, 11, 25, 23,
        16, 18, 10, 24, 14, 20, 17,
    ),
}
# fmt: on


def get_cheart_order_for_vtk(elem: VtkEnum) -> tuple[int, ...]:
    return Vtk2CheartNodeOrder[elem]
