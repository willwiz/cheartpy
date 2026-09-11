from collections import defaultdict
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
from pytools.arrays import A1

__all__ = [
    "dlagrange_1",
    "dlagrange_2",
    "lagrange_1",
    "lagrange_2",
]

type Vertex = tuple[int, int, int]
type ShapeFunction[T: np.floating] = Mapping[Vertex, A1[T]]


class ShapeFunctionDerivative[T: np.floating](NamedTuple):
    dx: ShapeFunction[T]
    dy: ShapeFunction[T]
    dz: ShapeFunction[T]


def lagrange_1[T: np.floating](x: A1[T]) -> ShapeFunction[T]:
    return {(0, 0, 0): 1 - x, (1, 0, 0): x}


def dlagrange_1[F: np.floating](x: A1[F]) -> ShapeFunctionDerivative[F]:
    return ShapeFunctionDerivative(
        {(0, 0, 0): np.full_like(x, -1), (1, 0, 0): np.full_like(x, 1)},
        defaultdict(lambda: np.zeros_like(x)),
        defaultdict(lambda: np.zeros_like(x)),
    )


def lagrange_2[F: np.floating](x: A1[F]) -> ShapeFunction[F]:
    return {
        (0, 0, 0): np.asarray((1 - x) * (1 - 2 * x), x.dtype),
        (1, 0, 0): np.asarray(4 * x * (1 - x), x.dtype),
        (2, 0, 0): np.asarray(x * (2 * x - 1), x.dtype),
    }


def dlagrange_2[F: np.floating](x: A1[F]) -> ShapeFunctionDerivative[F]:
    return ShapeFunctionDerivative(
        {
            (0, 0, 0): np.asarray(-(1 - 2 * x) - 2 * (1 - x), x.dtype),
            (1, 0, 0): np.asarray(4 * (1 - x) - 4 * x, x.dtype),
            (2, 0, 0): np.asarray((2 * x - 1) + x * 2, x.dtype),
        },
        defaultdict(lambda: np.zeros_like(x)),
        defaultdict(lambda: np.zeros_like(x)),
    )


def tri_lagrange_1[F: np.floating](x: A1[F], y: A1[F]) -> ShapeFunction[F]:
    return {(0, 0, 0): np.asarray(1.0 - x - y, x.dtype), (1, 0, 0): x, (0, 1, 0): y}


def dtri_lagrange_1[F: np.floating](x: A1[F], y: A1[F]) -> ShapeFunctionDerivative[F]:
    return ShapeFunctionDerivative(
        {
            (0, 0, 0): np.full_like(x, -1),
            (1, 0, 0): np.full_like(x, 1),
            (0, 1, 0): np.zeros_like(x),
        },
        {
            (0, 0, 0): np.full_like(y, -1),
            (1, 0, 0): np.zeros_like(y),
            (0, 1, 0): np.full_like(y, 1),
        },
        defaultdict(lambda: np.zeros_like(x)),
    )


def tri_lagrange_2[F: np.floating](x: A1[F], y: A1[F]) -> ShapeFunction[F]:
    return {
        (0, 0, 0): np.asarray((1 - x - y) * (1 - 2 * x - 2 * y), x.dtype),
        (2, 0, 0): np.asarray(x * (2 * x - 1), x.dtype),
        (0, 2, 0): np.asarray(y * (2 * y - 1), x.dtype),
        (1, 0, 0): np.asarray(4 * x * (1 - x - y), x.dtype),
        (0, 1, 0): np.asarray(4 * y * (1 - x - y), x.dtype),
        (1, 1, 0): np.asarray(4 * x * y, x.dtype),
    }


def dtri_lagrange_2[F: np.floating](x: A1[F], y: A1[F]) -> ShapeFunctionDerivative[F]:
    return ShapeFunctionDerivative(
        {
            (0, 0, 0): np.asarray((-1) * (1 - 2 * x - 2 * y) + (1 - x - y) * (-2), x.dtype),
            (2, 0, 0): np.asarray((2 * x - 1) + x * (2), x.dtype),
            (0, 2, 0): np.zeros_like(x),
            (1, 0, 0): np.asarray(4 * (1 - x - y) + 4 * x * (-1), x.dtype),
            (0, 1, 0): np.asarray(4 * y * (-1), x.dtype),
            (1, 1, 0): np.asarray(4 * y, x.dtype),
        },
        {
            (0, 0, 0): np.asarray((-1) * (1 - 2 * x - 2 * y) + (1 - x - y) * (-2), y.dtype),
            (2, 0, 0): np.zeros_like(y),
            (0, 2, 0): np.asarray((2 * y - 1) + y * (2), y.dtype),
            (1, 0, 0): np.asarray(4 * x * (-1), y.dtype),
            (0, 1, 0): np.asarray(4 * (1 - x - y) + 4 * y * (-1), y.dtype),
            (1, 1, 0): np.asarray(4 * x, y.dtype),
        },
        defaultdict(lambda: np.zeros_like(x)),
    )


def tet_lagrange_1[F: np.floating](x: A1[F], y: A1[F], z: A1[F]) -> ShapeFunction[F]:
    return {
        (0, 0, 0): np.asarray(1 - x - y - z, x.dtype),
        (1, 0, 0): x,
        (0, 1, 0): y,
        (0, 0, 1): z,
    }


def dtet_lagrange_1[F: np.floating](x: A1[F], y: A1[F], z: A1[F]) -> ShapeFunctionDerivative[F]:
    return ShapeFunctionDerivative(
        {
            (0, 0, 0): np.full_like(x, -1),
            (1, 0, 0): np.full_like(x, 1),
            (0, 1, 0): np.zeros_like(x),
            (0, 0, 1): np.zeros_like(x),
        },
        {
            (0, 0, 0): np.full_like(y, -1),
            (1, 0, 0): np.zeros_like(y),
            (0, 1, 0): np.full_like(y, 1),
            (0, 0, 1): np.zeros_like(y),
        },
        {
            (0, 0, 0): np.full_like(z, -1),
            (1, 0, 0): np.zeros_like(z),
            (0, 1, 0): np.zeros_like(z),
            (0, 0, 1): np.full_like(z, 1),
        },
    )


def tet_lagrange_2[F: np.floating](x: A1[F], y: A1[F], z: A1[F]) -> ShapeFunction[F]:
    return {
        (0, 0, 0): np.asarray((1 - x - y - z) * (1 - 2 * x - 2 * y - 2 * z), x.dtype),
        (2, 0, 0): np.asarray(x * (2 * x - 1), x.dtype),
        (0, 2, 0): np.asarray(y * (2 * y - 1), x.dtype),
        (0, 0, 2): np.asarray(z * (2 * z - 1), x.dtype),
        (1, 0, 0): np.asarray(4 * x * (1 - x - y - z), x.dtype),
        (0, 1, 0): np.asarray(4 * y * (1 - x - y - z), x.dtype),
        (0, 0, 1): np.asarray(4 * z * (1 - x - y - z), x.dtype),
        (1, 1, 0): np.asarray(4 * x * y, x.dtype),
        (1, 0, 1): np.asarray(4 * x * z, x.dtype),
        (0, 1, 1): np.asarray(4 * y * z, x.dtype),
    }


def dtet_lagrange_2[F: np.floating](x: A1[F], y: A1[F], z: A1[F]) -> ShapeFunctionDerivative[F]:
    return ShapeFunctionDerivative(
        {
            (0, 0, 0): np.asarray(
                (-1) * (1 - 2 * x - 2 * y - 2 * z) + (1 - x - y - z) * (-2), x.dtype
            ),
            (2, 0, 0): np.asarray((2 * x - 1) + x * (2), x.dtype),
            (0, 2, 0): np.zeros_like(x),
            (0, 0, 2): np.zeros_like(x),
            (1, 0, 0): np.asarray(4 * (1 - x - y - z) + 4 * x * (-1), x.dtype),
            (0, 1, 0): np.asarray(4 * y * (-1), x.dtype),
            (0, 0, 1): np.asarray(4 * z * (-1), x.dtype),
            (1, 1, 0): np.asarray(4 * y, x.dtype),
            (1, 0, 1): np.asarray(4 * z, x.dtype),
            (0, 1, 1): np.zeros_like(x),
        },
        {
            (0, 0, 0): np.asarray(
                (-1) * (1 - 2 * x - 2 * y - 2 * z) + (1 - x - y - z) * (-2), y.dtype
            ),
            (2, 0, 0): np.zeros_like(y),
            (0, 2, 0): np.asarray((2 * y - 1) + y * (2), y.dtype),
            (0, 0, 2): np.zeros_like(y),
            (1, 0, 0): np.asarray(4 * x * (-1), y.dtype),
            (0, 1, 0): np.asarray(4 * (1 - x - y - z) + 4 * y * (-1), y.dtype),
            (0, 0, 1): np.asarray(4 * z * (-1), y.dtype),
            (1, 1, 0): np.asarray(4 * x, y.dtype),
            (1, 0, 1): np.zeros_like(y),
            (0, 1, 1): np.asarray(4 * z, y.dtype),
        },
        {
            (0, 0, 0): np.asarray(
                (-1) * (1 - 2 * x - 2 * y - 2 * z) + (1 - x - y - z) * (-2), z.dtype
            ),
            (2, 0, 0): np.zeros_like(z),
            (0, 2, 0): np.zeros_like(z),
            (0, 0, 2): np.asarray((2 * z - 1) + z * (2), z.dtype),
            (1, 0, 0): np.asarray(4 * x * (-1), z.dtype),
            (0, 1, 0): np.asarray(4 * y * (-1), z.dtype),
            (0, 0, 1): np.asarray(4 * (1 - x - y - z) + 4 * z * (-1), z.dtype),
            (1, 1, 0): np.zeros_like(z),
            (1, 0, 1): np.asarray(4 * x, z.dtype),
            (0, 1, 1): np.asarray(4 * y, z.dtype),
        },
    )
