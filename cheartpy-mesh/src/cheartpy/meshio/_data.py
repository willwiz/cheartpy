from typing import TYPE_CHECKING

from cheartpy.vtk.types import VtkType

if TYPE_CHECKING:
    from collections.abc import Mapping

MESHIO_2_CHEART_CONNECTIVITY: Mapping[VtkType, Mapping[int, int]] = {
    VtkType.LinLine: {0: 0, 1: 1},
    VtkType.QuadLine: {0: 0, 1: 2, 2: 1},
    VtkType.LinTriangle: {0: 0, 1: 1, 2: 2},
    VtkType.QuadTriangle: {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 4},
    VtkType.LinQuadrilateral: {0: 0, 1: 1, 2: 3, 3: 2},
    VtkType.QuadQuadrilateral: {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 7, 6: 8, 7: 5, 8: 6},
    VtkType.LinTetrahedron: {0: 0, 1: 1, 2: 2, 3: 3},
    VtkType.QuadTetrahedron: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 5, 7: 7, 8: 8, 9: 9},
    VtkType.LinHexahedron: {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 7, 7: 6},
    VtkType.QuadHexahedron: {
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
CHEART_2_MESHIO_CONNECTIVITY: Mapping[VtkType, Mapping[int, int]] = {
    element: {v: k for k, v in conn.items()}
    for element, conn in MESHIO_2_CHEART_CONNECTIVITY.items()
}
