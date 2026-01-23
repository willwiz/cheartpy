from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pytools.arrays import A1, DType

__all__ = [
    "dlagrange_1",
    "dlagrange_2",
    "lagrange_1",
    "lagrange_2",
]


def lagrange_1[F: np.floating](x: float, *, dtype: DType[F] = np.float64) -> A1[F]:
    return np.array([1 - x, x], dtype=dtype)


def dlagrange_1[F: np.floating](_x: float, *, dtype: DType[F] = np.float64) -> A1[F]:
    return np.array([-1, 1], dtype=dtype)


def lagrange_2[F: np.floating](x: float, *, dtype: DType[F] = np.float64) -> A1[F]:
    return np.array([(1 - x) * (1 - 2 * x), 4 * x * (1 - x), x * (2 * x - 1)], dtype=dtype)


def dlagrange_2[F: np.floating](x: float, *, dtype: DType[F] = np.float64) -> A1[F]:
    return np.array([4 * x - 3, 4 - 8 * x, 4 * x - 1], dtype=dtype)


def tri_lagrange_1[F: np.floating](x: float, y: float, *, dtype: DType[F] = np.float64) -> A1[F]:
    return np.array([1.0 - x - y, x, y], dtype=dtype)


def dtri_lagrange_1[F: np.floating](_x: float, _y: float, *, dtype: DType[F] = np.float64) -> A1[F]:
    return np.array([[-1, 1, 0], [-1, 0, 1]], dtype=dtype)


def tri_lagrange_2[F: np.floating](x: float, y: float, *, dtype: DType[F] = np.float64) -> A1[F]:
    return np.array(
        [
            (1 - x - y) * (1 - 2 * (x + y)),
            x * (2 * x - 1),
            y * (2 * y - 1),
            4 * x * (1 - x - y),
            4 * x * y,
            4 * y * (1 - x - y),
        ],
        dtype=dtype,
    )


def dtri_lagrange_2[F: np.floating](x: float, y: float, *, dtype: DType[F] = np.float64) -> A1[F]:
    return np.array(
        [
            [4 * x + 4 * y - 3, 4 * x - 1, 4 * y],
            [4 * x + 4 * y - 3, 4 * y, 4 * x - 1],
        ],
        dtype=dtype,
    )
