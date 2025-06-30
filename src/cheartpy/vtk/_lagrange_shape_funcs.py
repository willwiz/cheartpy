__all__ = [
    "dlagrange_1",
    "dlagrange_2",
    "lagrange_1",
    "lagrange_2",
]

import numpy as np
from arraystubs import Arr1


def lagrange_1[F: np.floating](x: float, *, dtype: type[F] = np.float64) -> Arr1[F]:
    return np.array([1 - x, x], dtype=dtype)


def dlagrange_1[F: np.floating](_x: float, *, dtype: type[F] = np.float64) -> Arr1[F]:
    return np.array([-1, 1], dtype=dtype)


def lagrange_2[F: np.floating](x: float, *, dtype: type[F] = np.float64) -> Arr1[F]:
    return np.array([(1 - x) * (1 - 2 * x), 4 * x * (1 - x), x * (2 * x - 1)], dtype=dtype)


def dlagrange_2[F: np.floating](x: float, *, dtype: type[F] = np.float64) -> Arr1[F]:
    return np.array([4 * x - 3, 4 - 8 * x, 4 * x - 1], dtype=dtype)
