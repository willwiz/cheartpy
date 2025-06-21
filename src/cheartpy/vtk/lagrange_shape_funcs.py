__all__ = [
    "dlagrange_1",
    "dlagrange_2",
    "lagrange_1",
    "lagrange_2",
]
import numpy as np
from arraystubs import Arr1


def lagrange_1(x: float) -> Arr1[np.float64]:
    return np.array([1 - x, x], dtype=np.float64)


def dlagrange_1(_x: float) -> Arr1[np.float64]:
    return np.array([-1, 1], dtype=np.float64)


def lagrange_2(x: float) -> Arr1[np.float64]:
    return np.array([(1 - x) * (1 - 2 * x), 4 * x * (1 - x), x * (2 * x - 1)], dtype=np.float64)


def dlagrange_2(x: float) -> Arr1[np.float64]:
    return np.array([4 * x - 3, 4 - 8 * x, 4 * x - 1], dtype=np.float64)
