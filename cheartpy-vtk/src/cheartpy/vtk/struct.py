from typing import TYPE_CHECKING

import numpy as np

from ._elements import dlagrange_2, lagrange_2
from .types import VTK_TYPE, VtkElem, VtkType

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
    VtkType.LinLine,
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
    VtkType.QuadLine,
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
    )


VTKTRIANGLE1 = VtkElem(
    VtkType.LinTriangle,
    VtkType.LinTriangle,
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
    VtkType.QuadTriangle,
    VtkType.QuadLine,
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
    VtkType.LinQuadrilateral,
    VtkType.LinLine,
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


def _shape_tetrahedron_1[F: np.floating](pos: A1[F]) -> A1[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[2] < 0.0) | (pos[0] + pos[1] + pos[2] > 1.0):
        return np.zeros((4,), dtype=pos.dtype)
    raise NotImplementedError


def _shape_tetrahedron_1_deriv[F: np.floating](pos: A1[F]) -> A2[F]:
    if (pos[0] < 0.0) | (pos[1] < 0.0) | (pos[2] < 0.0) | (pos[0] + pos[1] + pos[2] > 1.0):
        return np.zeros((4, 4), dtype=pos.dtype)
    raise NotImplementedError


VTKTETRAHEDRON1 = VtkElem(
    VtkType.LinTetrahedron,
    VtkType.LinTriangle,
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
    _shape_hexahedron_2,
    _shape_hexahedron_2_deriv,
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
