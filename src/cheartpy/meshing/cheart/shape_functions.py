import numpy as np
from ...types import *


def sf_line_linear(coord: Vec[f64]) -> Vec[f64]:
    if coord[0] < 0.0 or coord[0] > 1.0:
        return np.zeros((2,), dtype=float)
    return np.ascontiguousarray([coord[0], 1.0 - coord[0]], dtype=float)


def dsf_line_linear(coord: Vec[f64]) -> Mat[f64]: ...


def sf_line_quadratic(coord: Vec[f64]) -> Vec[f64]:
    if coord[0] < 0.0 or coord[0] > 1.0:
        return np.zeros((3,), dtype=float)
    denom_arr = np.array([2.0, 4.0, 2.0], dtype=float)
    pos_arr = coord[0] - np.linspace(0.0, 1.0, 3, dtype=float)
    nom: Vec[f64] = np.ascontiguousarray(
        [np.prod(pos_arr[:i]) * np.prod(pos_arr[i + 1 :]) for i in range(len(pos_arr))],
        dtype=float,
    )
    return denom_arr * nom


def dsf_line_quadratic(coord: Vec[f64]) -> Mat[f64]: ...


def sf_triangle_linear(coord: Vec[f64]) -> Vec[f64]:
    if coord[0] < 0.0 or coord[1] < 0.0 or coord[0] + coord[1] > 1.0:
        return np.zeros((3,), dtype=float)
    return np.array([1.0 - coord[0] - coord[1], coord[0], coord[1]], dtype=float)


def dsf_triangle_linear(coord: Vec[f64]) -> Mat[f64]:
    if coord[0] < 0.0 or coord[1] < 0.0 or coord[0] + coord[1] > 1.0:
        return np.zeros((3,), dtype=float)
    return np.array(
        [
            [-1, 1, 0],
            [-1, 0, 1],
            [0, 0, 0],
        ],
        dtype=float,
    )


def sf_triangle_quadratic(coord: Vec[f64]) -> Vec[f64]: ...
def dsf_triangle_quadratic(coord: Vec[f64]) -> Mat[f64]: ...
def sf_quadrilateral_linear(coord: Vec[f64]) -> Vec[f64]: ...
def dsf_quadrilateral_linear(coord: Vec[f64]) -> Mat[f64]: ...
def sf_quadrilateral_quadratic(coord: Vec[f64]) -> Vec[f64]: ...
def dsf_quadrilateral_quadratic(coord: Vec[f64]) -> Mat[f64]: ...
def sf_tetrahedron_linear(coord: Vec[f64]) -> Vec[f64]: ...
def dsf_tetrahedron_linear(coord: Vec[f64]) -> Mat[f64]: ...
def sf_tetrahedron_quadratic(coord: Vec[f64]) -> Vec[f64]: ...
def dsf_tetrahedron_quadratic(coord: Vec[f64]) -> Mat[f64]: ...
def sf_hexahedron_linear(coord: Vec[f64]) -> Vec[f64]: ...
def dsf_hexahedron_linear(coord: Vec[f64]) -> Mat[f64]: ...
def sf_hexahedron_quadratic(coord: Vec[f64]) -> Vec[f64]: ...
def dsf_hexahedron_quadratic(coord: Vec[f64]) -> Mat[f64]: ...
