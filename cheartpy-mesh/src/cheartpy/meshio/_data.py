from typing import TYPE_CHECKING

from cheartpy.vtk.types import VtkEnum

if TYPE_CHECKING:
    from collections.abc import Mapping

MESHIO_2_CHEART_CONNECTIVITY: Mapping[VtkEnum, Mapping[int, int]] = {
    VtkEnum.VtkLinearLine: {0: 0, 1: 1},
    VtkEnum.VtkQuadraticLine: {0: 0, 1: 2, 2: 1},
    VtkEnum.VtkLinearTriangle: {0: 0, 1: 1, 2: 2},
    VtkEnum.VtkQuadraticTriangle: {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 4},
    VtkEnum.VtkLinearQuadrilateral: {0: 0, 1: 1, 2: 3, 3: 2},
    VtkEnum.VtkQuadraticQuadrilateral: {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 7, 6: 8, 7: 5, 8: 6},
    VtkEnum.VtkLinearTetrahedron: {0: 0, 1: 1, 2: 2, 3: 3},
    VtkEnum.VtkQuadraticTetrahedron: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 5, 7: 7, 8: 8, 9: 9},
    VtkEnum.VtkLinearHexahedron: {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 7, 7: 6},
    VtkEnum.VtkQuadraticHexahedron: {
        0: 0,
        1: 1,
        2: 3,
        3: 2,
        4: 4,
        5: 5,
        6: 7,
        7: 6,
        8: 8,
        9: 11,
        10: 24,
        11: 9,
        12: 10,
        13: 16,
        14: 22,
        15: 17,
        16: 20,
        17: 26,
        18: 21,
        19: 19,
        20: 23,
        21: 18,
        22: 12,
        23: 15,
        24: 25,
        25: 13,
        26: 14,
    },
}
CHEART_2_MESHIO_CONNECTIVITY: Mapping[VtkEnum, Mapping[int, int]] = {
    element: {v: k for k, v in conn.items()}
    for element, conn in MESHIO_2_CHEART_CONNECTIVITY.items()
}
