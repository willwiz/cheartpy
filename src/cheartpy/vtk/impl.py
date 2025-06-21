from __future__ import annotations

__all__ = ["get_vtk_elem"]

import numpy as np
from arraystubs import Arr1, Arr2

from cheartpy.vtk.lagrange_shape_funcs import dlagrange_2, lagrange_2

from .trait import VTK_TYPE, VtkElem, VtkType


def _shape_line_1[T: np.floating](pos: Arr1[T]) -> Arr1[np.float64]:
    if pos[0] < 0.0 or pos[0] > 1.0:
        return np.zeros((2,), dtype=np.float64)
    return np.array([1.0 - pos[0], pos[0]], dtype=np.float64)


def _shape_line_1_deriv[T: np.floating](pos: Arr1[T]) -> Arr2[np.float64]:
    if pos[0] < 0.0 or pos[0] > 1.0:
        return np.zeros((2, 3), dtype=np.float64)
    return np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)


VTKLINE1 = VtkElem(
    VtkType.LinLine,
    None,
    (0, 1),
    np.array([[0, 0, 0], [1, 0, 0]], dtype=np.intc),
    np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
    _shape_line_1,
    _shape_line_1_deriv,
)


def _shape_line_2[T: np.floating](pos: Arr1[T]) -> Arr1[np.float64]:
    if pos[0] < 0.0 or pos[0] > 1.0:
        return np.zeros((3,), dtype=np.float64)
    return np.array(
        [
            (1.0 - pos[0]) * (1.0 - 0.5 * pos[0]),
            pos[0] * (2.0 * pos[0] - 1.0),
            4.0 * pos[0] * (1.0 - pos[0]),
        ],
        dtype=np.float64,
    )


def _shape_line_2_deriv[T: np.floating](pos: Arr1[T]) -> Arr2[np.float64]:
    if pos[0] < 0.0 or pos[0] > 1.0:
        return np.zeros((3, 3), dtype=np.float64)
    return np.array(
        [
            [-3.0 + 4.0 * pos[0], 0.0, 0.0],
            [4.0 * pos[0] - 1.0, 0.0, 0.0],
            [4.0 - 8.0 * pos[0], 0.0, 0.0],
        ],
        dtype=np.float64,
    )


VTKLINE2 = VtkElem(
    VtkType.QuadLine,
    None,
    (0, 1, 2),
    np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0]], dtype=np.intc),
    np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], dtype=np.float64),
    _shape_line_2,
    _shape_line_2_deriv,
)


VTKTRIANGLE1 = VtkElem(
    VtkType.LinTriangle,
    VtkType.LinTriangle,
    (0, 1, 2),
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.intc),
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
    lambda pos: np.where(
        (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[0] + pos[1] > 1.0),
        np.zeros(3, dtype=pos.dtype),
        np.array(
            [[1.0 - pos[0] - pos[1], pos[0], pos[1]]],
            dtype=pos.dtype,
        ).T,
    ),
    lambda pos: np.where(
        (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[0] + pos[1] > 1.0),
        np.zeros((3, 3), dtype=pos.dtype),
        np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=pos.dtype,
        ),
    ),
)


VTKTRIANGLE2 = VtkElem(
    VtkType.QuadTriangle,
    VtkType.QuadLine,
    (0, 1, 2, 3, 5, 4),
    np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.intc),
    np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]],
        dtype=np.float64,
    ),
    lambda pos: np.where(
        (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[0] + pos[1] > 1.0),
        np.zeros(6, dtype=pos.dtype),
        np.array(
            [
                (1 - pos[0] - pos[1]) * (1 - 2 * pos[0] - 2 * pos[1]),
                pos[0] * (2 * pos[0] - 1),
                pos[1] * (2 * pos[1] - 1),
                4 * pos[0] * (1 - pos[0] - pos[1]),
                4 * pos[1] * (1 - pos[0] - pos[1]),
                4 * pos[0] * pos[1],
            ],
            dtype=pos.dtype,
        ),
    ),
    lambda pos: np.where(
        (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[0] + pos[1] > 1.0),
        np.zeros((6, 3), dtype=pos.dtype),
        np.array(
            [
                [
                    -3.0 + 4.0 * pos[0] + 4.0 * pos[1],
                    -3.0 + 4.0 * pos[0] + 4.0 * pos[1],
                    0.0,
                ],
                [-1 + 4.0 * pos[0], 0.0, 0.0],
                [0.0, -1 + 4.0 * pos[1], 0.0],
                [
                    4.0 * pos[0] * (-1 + 2 * pos[0] + pos[1]),
                    4.0 * pos[0],
                    0.0,
                ],
                [
                    4.0 * pos[1],
                    4.0 * pos[1] * (-1 + 2 * pos[0] + pos[1]),
                    0.0,
                ],
                [4.0 * pos[1], 4.0 * pos[0], 0.0],
            ],
            dtype=pos.dtype,
        ),
    ),
)


VTKQUADRILATERAL1 = VtkElem(
    VtkType.LinQuadrilateral,
    VtkType.LinLine,
    (0, 1, 3, 2),
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.intc),
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64),
    lambda pos: np.where(
        (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[0] > 1.0) | (pos[1] > 1.0),
        np.zeros(4, dtype=pos.dtype),
        np.array(
            [
                (1.0 - pos[0]) * (1.0 - pos[1]),
                pos[0] * (1.0 - pos[1]),
                (1.0 - pos[0]) * pos[1],
                pos[0] * pos[1],
            ],
            dtype=pos.dtype,
        ),
    ),
    lambda pos: np.where(
        (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[0] > 1.0) | (pos[1] > 1.0),
        np.zeros((4, 3), dtype=pos.dtype),
        np.array(
            [
                [-1.0 + pos[1], -1.0 + pos[0], 0.0],
                [1.0 - pos[1], -pos[0], 0.0],
                [-pos[1], 1.0 - pos[0], 0.0],
                [pos[1], pos[0], 0.0],
            ],
            dtype=pos.dtype,
        ),
    ),
)


def _shape_quad_2[T: np.floating](pos: Arr1[T]) -> Arr1[np.float64]:
    if pos[0] < 0.0 or pos[1] < 0.0 or pos[0] > 1.0 or pos[1] > 1.0:
        return np.zeros((9,), dtype=np.float64)
    dx = lagrange_2(pos[0])
    dy = lagrange_2(pos[1])
    return np.array(
        [
            dx[0] * dy[0],
            dx[2] * dy[0],
            dx[0] * dy[2],
            dx[2] * dy[2],
            dx[1] * dy[0],
            dx[0] * dy[1],
            dx[1] * dy[1],
            dx[2] * dy[1],
            dx[1] * dy[2],
        ],
        dtype=np.float64,
    )


def _shape_quad_2_deriv[T: np.floating](pos: Arr1[T]) -> Arr2[np.float64]:
    if pos[0] < 0.0 or pos[1] < 0.0 or pos[0] > 1.0 or pos[1] > 1.0:
        return np.zeros((9, 3), dtype=np.float64)
    dx = lagrange_2(pos[0])
    dxdt = dlagrange_2(pos[0])
    dy = lagrange_2(pos[1])
    dydt = dlagrange_2(pos[1])
    return np.array(
        [
            [
                dxdt[0] * dy[0],
                dxdt[2] * dy[0],
                dxdt[0] * dy[2],
                dxdt[2] * dy[2],
                dxdt[1] * dy[0],
                dxdt[0] * dy[1],
                dxdt[1] * dy[1],
                dxdt[2] * dy[1],
                dxdt[1] * dy[2],
            ],
            [
                dx[0] * dydt[0],
                dx[2] * dydt[0],
                dx[0] * dydt[2],
                dx[2] * dydt[2],
                dx[1] * dydt[0],
                dx[0] * dydt[1],
                dx[1] * dydt[1],
                dx[2] * dydt[1],
                dx[1] * dydt[2],
            ],
            [0.0] * 9,
        ],
        dtype=np.float64,
    )


VTKQUADRILATERAL2 = VtkElem(
    VtkType.QuadQuadrilateral,
    VtkType.QuadLine,
    (0, 1, 3, 2, 4, 7, 8, 5, 6),
    np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
            [2, 2, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],
            [1, 2, 0],
        ],
        dtype=np.intc,
    ),
    np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0.5, 0.5, 0],
            [1, 0.5, 0],
            [0.5, 1, 0],
        ],
        dtype=np.float64,
    ),
    _shape_quad_2,
    _shape_quad_2_deriv,
)

# NOTE: NOT IMPLEMENTED
VTKTETRAHEDRON1 = VtkElem(
    VtkType.LinTetrahedron,
    VtkType.LinTriangle,
    (0, 1, 2, 3),
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.intc),
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    lambda pos: np.where(
        (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[2] < 0.0) | (pos[0] + pos[1] + pos[2] > 1.0),
        np.zeros(4, dtype=pos.dtype),
        np.array(
            [[1.0 - pos[0] - pos[1] - pos[2], pos[0], pos[1], pos[2]]],
            dtype=pos.dtype,
        ).T,
    ),
    lambda pos: np.where(
        (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[2] < 0.0) | (pos[0] + pos[1] + pos[2] > 1.0),
        np.zeros((4, 4), dtype=pos.dtype),
        np.array(
            [
                [-1.0, -1.0, -1.0, 3.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 3.5],
            ],
            dtype=pos.dtype,
        ),
    ),
)

# NOTE: NOT IMPLEMENTED
VTKTETRAHEDRON2 = VtkElem(
    VtkType.QuadTetrahedron,
    VtkType.QuadTriangle,
    (0, 1, 2, 3, 4, 6, 5, 7, 8, 9),
    np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.intc,
    ),
    np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0.5, 0.5, 0],
            [0, 0, 0.5],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
        ],
        dtype=np.float64,
    ),
    lambda pos: np.where(
        (pos[0] < 0.0)
        | (pos[1] < 0.0)
        | (pos[2] < 0.0)
        | (pos[3] < 0.0)
        | (pos[4] + pos[5] + pos[6] > pos[7]),
        np.zeros(9, dtype=pos.dtype),
        np.array(
            [
                (1 - pos[4]) * (1 - pos[5]) * (1 - pos[6]),
                pos[4] * (2 * pos[4] - pos[7]),
                pos[5] * (2 * pos[5] - pos[8]),
                pos[6] * (2 * pos[6] - pos[9]),
                (pos[4] + pos[5]) * (pos[6] + pos[7]),
                (pos[4] + pos[5]) * (pos[6] + pos[7]),
                (pos[4] + pos[5]) * (pos[6] + pos[7]),
                (pos[4] + pos[5]) * (pos[6] + pos[7]),
                (1 - pos[4]) * (1 - pos[5]) * (1 - pos[6]),
            ],
            dtype=pos.dtype,
        ),
    ),
    lambda pos: np.where(
        (pos[0] < 0.0)
        | (pos[1] < 0.0)
        | (pos[2] < 0.0)
        | (pos[3] < 0.0)
        | (pos[4] + pos[5] + pos[6] > pos[7]),
        np.zeros((4, 9), dtype=pos.dtype),
        np.array(
            [
                [
                    -3.0 + 4.0 * pos[4],
                    -3.0 + 4.0 * pos[5],
                    -3.0 + 4.0 * pos[6],
                    4.0 * pos[7] - 1.0,
                    4.0 * pos[8] - 1.0,
                    4.0 * pos[9] - 1.0,
                    4.0 * pos[10] - 1.0,
                    4.0 * pos[11] - 1.0,
                    4.0 * pos[12] - 1.0,
                ],
                [
                    4.0 * pos[7] - 1.0,
                    4.0 * pos[8],
                    4.0 - 8.0 * pos[8],
                    -2 + 8 * pos[8],
                    -2 + 8 * pos[9],
                    -2 + 8 * pos[10],
                    -2 + 8 * pos[11],
                    -2 + 8 * pos[12],
                    -2 + 8 * pos[13],
                ],
                [
                    4.0 - 8.0 * pos[4],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    4.0 - 8.0 * pos[5],
                ],
                [
                    4.0 - 8.0 * pos[6],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    4.0 - 8.0 * pos[7],
                ],
            ],
            dtype=pos.dtype,
        ),
    ),
)
# NOTE: NOT IMPLEMENTED
VTKHEXAHEDRON1 = VtkElem(
    VtkType.LinHexahedron,
    VtkType.LinQuadrilateral,
    (0, 1, 5, 4, 2, 3, 7, 6),
    np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.intc,
    ),
    np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    ),
    lambda pos: np.where(
        (pos[0] < 0.0)
        | (pos[1] < 0.0)
        | (pos[2] < 0.0)
        | (pos[3] < 0.0)
        | (pos[4] > pos[5])
        | (pos[6] > pos[7]),
        np.zeros(8, dtype=pos.dtype),
        np.array(
            [
                (1 - pos[4]) * (1 - pos[5]) * (1 - pos[6]) * (1 - pos[7]),
                pos[4] * (2 * pos[4] - pos[8]),
                pos[5] * (2 * pos[5] - pos[9]),
                pos[6] * (2 * pos[6] - pos[10]),
                (pos[4] + pos[5]) * (pos[6] + pos[7]),
                (pos[4] + pos[5]) * (pos[6] + pos[7]),
                (pos[4] + pos[5]) * (pos[6] + pos[7]),
                (pos[4] + pos[5]) * (pos[6] + pos[7]),
            ],
            dtype=pos.dtype,
        ),
    ),
    lambda pos: np.where(
        (pos[0] < 0.0)
        | (pos[1] < 0.0)
        | (pos[2] < 0.0)
        | (pos[3] < 0.0)
        | (pos[4] > pos[5])
        | (pos[6] > pos[7]),
        np.zeros((3, 8), dtype=pos.dtype),
        np.array(
            [
                [
                    -3.0 + 4.0 * pos[4],
                    -3.0 + 4.0 * pos[5],
                    -3.0 + 4.0 * pos[6],
                    -3.0 + 4.0 * pos[7],
                    4.0 * pos[8] - 1.0,
                    4.0 * pos[9] - 1.0,
                    4.0 * pos[10] - 1.0,
                    4.0 * pos[11] - 1.0,
                ],
                [
                    4.0 * pos[8] - 1.0,
                    4.0 * pos[9],
                    4.0 - 8.0 * pos[9],
                    -2 + 8 * pos[9],
                    -2 + 8 * pos[10],
                    -2 + 8 * pos[11],
                    -2 + 8 * pos[12],
                    -2 + 8 * pos[13],
                ],
                [
                    4.0 - 8.0 * pos[4],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    4.0 - 8.0 * pos[5],
                    4.0 - 8.0 * pos[6],
                ],
            ],
            dtype=pos.dtype,
        ),
    ),
)

# NOTE: NOT IMPLEMENTED
VTKHEXAHEDRON2 = VtkElem(
    VtkType.QuadHexahedron,
    VtkType.QuadQuadrilateral,
    (
        0,  1,  5,  4,  2,  3,  7, 6,  8,  15,
        22, 13, 12, 21, 26, 19, 9, 11, 25, 23,
        16, 18, 10, 24, 14, 20, 17,
    ),
    np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
            [2, 2, 0],
            [0, 0, 2],
            [2, 0, 2],
            [0, 2, 2],
            [2, 2, 2],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],
            [1, 2, 0],
            [0, 0, 1],
            [1, 0, 1],
            [2, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [2, 1, 1],
            [0, 2, 1],
            [1, 2, 1],
            [2, 2, 1],
            [1, 0, 2],
            [0, 1, 2],
            [1, 1, 2],
            [2, 1, 2],
            [1, 2, 2],
        ],
        dtype=np.intc,
    ),
    np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0.5, 0.5, 0],
            [1, 0.5, 0],
            [0.5, 1, 0],
            [0, 0, 0.5],
            [0.5, 0, 0.5],
            [1, 0, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [1, 0.5, 0.5],
            [0, 1, 0.5],
            [0.5, 1, 0.5],
            [1, 1, 0.5],
            [0.5, 0, 1],
            [0, 0.5, 1],
            [0.5, 0.5, 1],
            [1, 0.5, 1],
            [0.5, 1, 1],
        ],
        dtype=np.float64,
    ),
    lambda pos: np.where(
        (pos[0] < 0.0)
        | (pos[1] < 0.0)
        | (pos[2] < 0.0)
        | (pos[3] < 0.0)
        | (pos[4] > pos[5])
        | (pos[6] > pos[7]),
        np.zeros(16, dtype=pos.dtype),
        np.array(
            [
                (1 - pos[8]) * (1 - pos[9]) * (1 - pos[10]) * (1 - pos[11]),
                pos[12] * (2 * pos[12] - pos[13]),
                pos[14] * (2 * pos[14] - pos[15]),
                pos[16] * (2 * pos[16] - pos[17]),
                (pos[18] + pos[19]) * (pos[20] + pos[21]),
                (pos[22] + pos[23]) * (pos[24] + pos[25]),
                (pos[26] + pos[27]) * (pos[28] + pos[29]),
                (pos[30] + pos[31]) * (pos[32] + pos[33]),
            ],
            dtype=pos.dtype,
        ),
    ),
    lambda pos: np.where(
        (pos[0] < 0.0)
        | (pos[1] < 0.0)
        | (pos[2] < 0.0)
        | (pos[3] < 0.0)
        | (pos[4] > pos[5])
        | (pos[6] > pos[7]),
        np.zeros((3, 16), dtype=pos.dtype),
        np.array(
            [
                [
                    -3.0 + 4.0 * pos[8],
                    -3.0 + 4.0 * pos[9],
                    -3.0 + 4.0 * pos[10],
                    -3.0 + 4.0 * pos[11],
                    4.0 * pos[12] - 1.0,
                    4.0 * pos[13] - 1.0,
                    4.0 * pos[14] - 1.0,
                    4.0 * pos[15] - 1.0,
                ],
                [
                    4.0 * pos[16] - 1.0,
                    4.0 * pos[17],
                    4.0 - 8.0 * pos[17],
                    -2 + 8 * pos[17],
                    -2 + 8 * pos[18],
                    -2 + 8 * pos[19],
                    -2 + 8 * pos[20],
                    -2 + 8 * pos[21],
                ],
                [
                    4.0 - 8.0 * pos[22],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    4.0 - 8.0 * pos[23],
                    4.0 - 8.0 * pos[24],
                    4.0 - 8.0 * pos[25],
                ],
            ],
            dtype=pos.dtype,
        ),
    ),
)  # fmt: skip


def get_vtk_elem(elem_type: VTK_TYPE | VtkType) -> VtkElem:
    if not isinstance(elem_type, VtkType):
        elem_type = VtkType[elem_type]
    elements = {
        VtkType.LinLine: VTKLINE1,
        VtkType.LinTriangle: VTKTRIANGLE1,
        VtkType.LinQuadrilateral: VTKQUADRILATERAL1,
        VtkType.LinTetrahedron: VTKTETRAHEDRON1,
        VtkType.LinHexahedron: VTKHEXAHEDRON1,
        VtkType.QuadLine: VTKLINE2,
        VtkType.QuadTriangle: VTKTRIANGLE2,
        VtkType.QuadQuadrilateral: VTKQUADRILATERAL2,
        VtkType.QuadTetrahedron: VTKTETRAHEDRON2,
        VtkType.QuadHexahedron: VTKHEXAHEDRON2,
    }
    return elements[elem_type]
