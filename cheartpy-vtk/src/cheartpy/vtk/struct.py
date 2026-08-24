from typing import TYPE_CHECKING

import numpy as np
from cheartpy.elem_interfaces import VtkElemType, VtkEnum

from ._elements import dlagrange_2, lagrange_2
from .types import VtkElem

if TYPE_CHECKING:
    from pytools.arrays import A1, A2

__all__ = [
    "VTKHEXAHEDRON1",
    "VTKHEXAHEDRON2",
    "VTKLINE1",
    "VTKLINE2",
    "VTKQUADRILATERAL1",
    "VTKQUADRILATERAL2",
    "VTKTETRAHEDRON1",
    "VTKTETRAHEDRON2",
    "VTKTRIANGLE1",
    "VTKTRIANGLE2",
]


def _shape_line_1[T: np.floating](pos: A1[T]) -> A1[T]:
    if pos[0] < 0.0 or pos[0] > 1.0:
        return np.zeros((2,), dtype=pos.dtype)
    return np.array([1.0 - pos[0], pos[0]], dtype=pos.dtype)


def _shape_line_1_deriv[T: np.floating](pos: A1[T]) -> A2[T]:
    if pos[0] < 0.0 or pos[0] > 1.0:
        return np.zeros((2, 3), dtype=pos.dtype)
    return np.array([[-1, 0, 0], [1, 0, 0]], dtype=pos.dtype).T


VTKLINE1 = VtkElem(
    VtkEnum.LINE1,
    None,
    (0, 1),
    np.array([[0, 0, 0], [1, 0, 0]], dtype=np.intc),
    np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
    _shape_line_1,
    _shape_line_1_deriv,
)


def _shape_line_2[T: np.floating](pos: A1[T]) -> A1[T]:
    if pos[0] < 0.0 or pos[0] > 1.0:
        return np.zeros((3,), dtype=pos.dtype)
    return np.array(
        [
            (1.0 - pos[0]) * (1.0 - 0.5 * pos[0]),
            pos[0] * (2.0 * pos[0] - 1.0),
            4.0 * pos[0] * (1.0 - pos[0]),
        ],
        dtype=pos.dtype,
    )


def _shape_line_2_deriv[T: np.floating](pos: A1[T]) -> A2[T]:
    if pos[0] < 0.0 or pos[0] > 1.0:
        return np.zeros((3, 3), dtype=pos.dtype)
    return np.array(
        [
            [-3.0 + 4.0 * pos[0], 0.0, 0.0],
            [4.0 * pos[0] - 1.0, 0.0, 0.0],
            [4.0 - 8.0 * pos[0], 0.0, 0.0],
        ],
        dtype=pos.dtype,
    )


VTKLINE2 = VtkElem(
    VtkEnum.LINE2,
    None,
    (0, 1, 2),
    np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0]], dtype=np.intc),
    np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], dtype=np.float64),
    _shape_line_2,
    _shape_line_2_deriv,
)


def _shape_triangle_1[F: np.floating](pos: A1[F]) -> A1[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[0] + pos[1] > 1.0):
        return np.zeros((3,), dtype=pos.dtype)
    return np.array(
        [[1.0 - pos[0] - pos[1], pos[0], pos[1]]],
        dtype=pos.dtype,
    )


def _shape_triangle_1_deriv[F: np.floating](pos: A1[F]) -> A2[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[0] + pos[1] > 1.0):
        return np.zeros((3, 3), dtype=pos.dtype)
    return np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=pos.dtype,
    ).T


VTKTRIANGLE1 = VtkElem(
    VtkEnum.TRIANGLE1,
    VtkEnum.LINE1,
    (0, 1, 2),
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.intc),
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
    _shape_triangle_1,
    _shape_triangle_1_deriv,
)


def _shape_triangle_2[F: np.floating](pos: A1[F]) -> A1[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[0] + pos[1] > 1.0):
        return np.zeros(6, dtype=pos.dtype)
    return np.array(
        [
            (1 - pos[0] - pos[1]) * (1 - 2 * pos[0] - 2 * pos[1]),
            pos[0] * (2 * pos[0] - 1),
            pos[1] * (2 * pos[1] - 1),
            4 * pos[0] * (1 - pos[0] - pos[1]),
            4 * pos[1] * (1 - pos[0] - pos[1]),
            4 * pos[0] * pos[1],
        ],
        dtype=pos.dtype,
    )


def _shape_triangle_2_deriv[F: np.floating](pos: A1[F]) -> A2[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[0] + pos[1] > 1.0):
        return np.zeros((6, 3), dtype=pos.dtype)
    return np.array(
        [
            [-3.0 + 4.0 * pos[0] + 4.0 * pos[1], -3.0 + 4.0 * pos[0] + 4.0 * pos[1], 0.0],
            [-1 + 4.0 * pos[0], 0.0, 0.0],
            [0.0, -1 + 4.0 * pos[1], 0.0],
            [4.0 * pos[0] * (-1 + 2 * pos[0] + pos[1]), 4.0 * pos[0], 0.0],
            [4.0 * pos[1], 4.0 * pos[1] * (-1 + 2 * pos[0] + pos[1]), 0.0],
            [4.0 * pos[1], 4.0 * pos[0], 0.0],
        ],
        dtype=pos.dtype,
    )


VTKTRIANGLE2 = VtkElem(
    VtkEnum.TRIANGLE2,
    VtkEnum.LINE2,
    (0, 1, 2, 3, 5, 4),
    np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.intc),
    np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]],
        dtype=np.float64,
    ),
    _shape_triangle_2,
    _shape_triangle_2_deriv,
)


def _shape_quad_1[F: np.floating](pos: A1[F]) -> A1[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[0] > 1.0) | (pos[1] > 1.0):
        return np.zeros((4,), dtype=pos.dtype)
    return np.array(
        [
            (1.0 - pos[0]) * (1.0 - pos[1]),
            pos[0] * (1.0 - pos[1]),
            (1.0 - pos[0]) * pos[1],
            pos[0] * pos[1],
        ],
        dtype=pos.dtype,
    )


def _shape_quad_1_deriv[F: np.floating](pos: A1[F]) -> A2[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[0] > 1.0) | (pos[1] > 1.0):
        return np.zeros((4, 3), dtype=pos.dtype)
    return np.array(
        [
            [-1.0 + pos[1], -1.0 + pos[0], 0.0],
            [1.0 - pos[1], -pos[0], 0.0],
            [-pos[1], 1.0 - pos[0], 0.0],
            [pos[1], pos[0], 0.0],
        ],
        dtype=pos.dtype,
    ).T


VTKQUADRILATERAL1 = VtkElem(
    VtkEnum.QUADRILATERAL1,
    VtkEnum.LINE1,
    (0, 1, 3, 2),
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.intc),
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64),
    _shape_quad_1,
    _shape_quad_1_deriv,
)


def _shape_quad_2[T: np.floating](pos: A1[T]) -> A1[T]:
    if pos[0] < 0.0 or pos[1] < 0.0 or pos[0] > 1.0 or pos[1] > 1.0:
        return np.zeros((9,), dtype=pos.dtype)
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
        dtype=pos.dtype,
    )


def _shape_quad_2_deriv[T: np.floating](pos: A1[T]) -> A2[T]:
    if pos[0] < 0.0 or pos[1] < 0.0 or pos[0] > 1.0 or pos[1] > 1.0:
        return np.zeros((9, 3), dtype=pos.dtype)
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
        dtype=pos.dtype,
    )


VTKQUADRILATERAL2 = VtkElem(
    VtkEnum.QUADRILATERAL2,
    VtkEnum.LINE2,
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


def _shape_tetrahedron_1[F: np.floating](pos: A1[F]) -> A1[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[2] < 0.0) | (pos[0] + pos[1] + pos[2] > 1.0):
        return np.zeros((4,), dtype=pos.dtype)
    raise NotImplementedError


def _shape_tetrahedron_1_deriv[F: np.floating](pos: A1[F]) -> A2[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[2] < 0.0) | (pos[0] + pos[1] + pos[2] > 1.0):
        return np.zeros((4, 4), dtype=pos.dtype)
    raise NotImplementedError


VTKTETRAHEDRON1 = VtkElem(
    VtkEnum.TETRAHEDRON1,
    VtkEnum.TRIANGLE1,
    (0, 1, 2, 3),
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.intc),
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    _shape_tetrahedron_1,
    _shape_tetrahedron_1_deriv,
)


def _shape_tetrahedron_2[F: np.floating](pos: A1[F]) -> A1[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[2] < 0.0) | (pos[3] < 0.0):
        return np.zeros((10,), dtype=pos.dtype)
    raise NotImplementedError


def _shape_tetrahedron_2_deriv[F: np.floating](pos: A1[F]) -> A2[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[2] < 0.0) | (pos[3] < 0.0):
        return np.zeros((10, 4), dtype=pos.dtype)
    raise NotImplementedError


VTKTETRAHEDRON2 = VtkElem(
    VtkEnum.TETRAHEDRON2,
    VtkEnum.TRIANGLE2,
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
    _shape_tetrahedron_2,
    _shape_tetrahedron_2_deriv,
)


def _shape_hexahedron_1[F: np.floating](pos: A1[F]) -> A1[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[2] < 0.0) | (pos[3] < 0.0):
        return np.zeros((8,), dtype=pos.dtype)
    raise NotImplementedError


def _shape_hexahedron_1_deriv[F: np.floating](pos: A1[F]) -> A2[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[2] < 0.0) | (pos[3] < 0.0):
        return np.zeros((8, 4), dtype=pos.dtype)
    raise NotImplementedError


VTKHEXAHEDRON1 = VtkElem(
    VtkEnum.HEXAHEDRON1,
    VtkEnum.QUADRILATERAL1,
    (0, 1, 3, 2, 4, 5, 7, 6),
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
    _shape_hexahedron_1,
    _shape_hexahedron_1_deriv,
)


def _shape_hexahedron_2[F: np.floating](pos: A1[F]) -> A1[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[2] < 0.0) | (pos[3] < 0.0):
        return np.zeros((16,), dtype=pos.dtype)
    raise NotImplementedError


def _shape_hexahedron_2_deriv[F: np.floating](pos: A1[F]) -> A2[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[2] < 0.0) | (pos[3] < 0.0):
        return np.zeros((16, 4), dtype=pos.dtype)
    raise NotImplementedError


VTKHEXAHEDRON2 = VtkElem(
    VtkEnum.HEXAHEDRON2,
    VtkEnum.QUADRILATERAL2,
    (
        0,   1,  3,  2,  4,  5,  7,  6,  8, 11,
        24,  9, 10, 16, 22, 17, 20, 26, 21, 19,
        23, 18, 12, 15, 26, 13, 15
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
    _shape_hexahedron_2,
    _shape_hexahedron_2_deriv,
)  # fmt: skip


def get_vtk_elem(elem_type: VtkElemType | VtkEnum) -> VtkElem:
    if not isinstance(elem_type, VtkEnum):
        elem_type = VtkEnum[elem_type]
    elements = {
        VtkEnum.LINE1: VTKLINE1,
        VtkEnum.TRIANGLE1: VTKTRIANGLE1,
        VtkEnum.QUADRILATERAL1: VTKQUADRILATERAL1,
        VtkEnum.TETRAHEDRON1: VTKTETRAHEDRON1,
        VtkEnum.HEXAHEDRON1: VTKHEXAHEDRON1,
        VtkEnum.LINE2: VTKLINE2,
        VtkEnum.TRIANGLE2: VTKTRIANGLE2,
        VtkEnum.QUADRILATERAL2: VTKQUADRILATERAL2,
        VtkEnum.TETRAHEDRON2: VTKTETRAHEDRON2,
        VtkEnum.HEXAHEDRON2: VTKHEXAHEDRON2,
    }
    return elements[elem_type]
