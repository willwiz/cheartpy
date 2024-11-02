__all__ = [
    "Lagrange_1",
    "Lagrange_2",
    "dLagrange_1",
    "dLagrange_2",
]
import numpy as np
from ..var_types import *


def Lagrange_1(x: float) -> Vec[f64]:
    return np.array([1 - x, x], dtype=float)


def dLagrange_1(x: float) -> Vec[f64]:
    return np.array([-1, 1], dtype=float)


def Lagrange_2(x: float) -> Vec[f64]:
    return np.array(
        [(1 - x) * (1 - 2 * x), 4 * x * (1 - x), x * (2 * x - 1)], dtype=float
    )


def dLagrange_2(x: float) -> Vec[f64]:
    return np.array([4 * x - 3, 4 - 8 * x, 4 * x - 1], dtype=float)
