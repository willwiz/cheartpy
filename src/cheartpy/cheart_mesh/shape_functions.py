from __future__ import annotations

__all__ = [
    "dsf_hexahedron_linear",
    "dsf_hexahedron_quadratic",
    "dsf_line_linear",
    "dsf_line_quadratic",
    "dsf_quadrilateral_linear",
    "dsf_quadrilateral_quadratic",
    "dsf_tetrahedron_linear",
    "dsf_tetrahedron_quadratic",
    "dsf_triangle_linear",
    "dsf_triangle_quadratic",
    "sf_hexahedron_linear",
    "sf_hexahedron_quadratic",
    "sf_line_linear",
    "sf_line_quadratic",
    "sf_quadrilateral_linear",
    "sf_quadrilateral_quadratic",
    "sf_tetrahedron_linear",
    "sf_tetrahedron_quadratic",
    "sf_triangle_linear",
    "sf_triangle_quadratic",
]
import numpy as np

from ..vtk.lagrange_shape_funcs import dlagrange_2, lagrange_2


def sf_triangle_quadratic(coord: Vec[f64]) -> Vec[f64]: ...
def dsf_triangle_quadratic(coord: Vec[f64]) -> Mat[f64]: ...
def sf_quadrilateral_linear(coord: Vec[f64]) -> Vec[f64]:
    if coord[0] < 0.0 or coord[1] < 0.0 or coord[0] > 1.0 or coord[1] > 1.0:
        return np.zeros((4,), dtype=float)
    return np.array(
        [
            (1.0 - coord[0]) * (1.0 - coord[1]),
            (coord[0]) * (1.0 - coord[1]),
            (1.0 - coord[0]) * (coord[1]),
            (coord[0]) * (coord[1]),
        ],
        dtype=float,
    )


def dsf_quadrilateral_linear(coord: Vec[f64]) -> Mat[f64]:
    if coord[0] < 0.0 or coord[1] < 0.0 or coord[0] > 1.0 or coord[1] > 1.0:
        return np.zeros((3, 4), dtype=float)
    return np.array(
        [
            [-(1.0 - coord[1]), 1.0 - coord[1], -coord[1], coord[1]],
            [-(1.0 - coord[0]), -coord[0], 1.0 - coord[0], coord[0]],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )


def sf_quadrilateral_quadratic(coord: Vec[f64]) -> Vec[f64]:
    if coord[0] < 0.0 or coord[1] < 0.0 or coord[0] > 1.0 or coord[1] > 1.0:
        return np.zeros((9,), dtype=float)
    dx = lagrange_2(coord[0])
    dy = lagrange_2(coord[1])
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
        dtype=float,
    )


def dsf_quadrilateral_quadratic(coord: Vec[f64]) -> Mat[f64]:
    if coord[0] < 0.0 or coord[1] < 0.0 or coord[0] > 1.0 or coord[1] > 1.0:
        return np.zeros((3, 9), dtype=float)
    dx = lagrange_2(coord[0])
    dxdt = dlagrange_2(coord[0])
    dy = lagrange_2(coord[1])
    dydt = dlagrange_2(coord[1])
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
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )


def sf_tetrahedron_linear(coord: Vec[f64]) -> Vec[f64]: ...
def dsf_tetrahedron_linear(coord: Vec[f64]) -> Mat[f64]: ...
def sf_tetrahedron_quadratic(coord: Vec[f64]) -> Vec[f64]: ...
def dsf_tetrahedron_quadratic(coord: Vec[f64]) -> Mat[f64]: ...
def sf_hexahedron_linear(coord: Vec[f64]) -> Vec[f64]: ...
def dsf_hexahedron_linear(coord: Vec[f64]) -> Mat[f64]: ...
def sf_hexahedron_quadratic(coord: Vec[f64]) -> Vec[f64]: ...
def dsf_hexahedron_quadratic(coord: Vec[f64]) -> Mat[f64]: ...
