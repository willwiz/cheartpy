from typing import TYPE_CHECKING

import numpy as np

from ._types import VtkElem, VtkType

if TYPE_CHECKING:
    from pytools.arrays import A1, A2


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
