__all__ = ["create_mesh"]
import numpy as np
from ...var_types import *
from ...cheart_mesh import *


def _create_square_nodal_index(nx: int, ny: int):
    index = np.zeros((nx + 1, ny + 1), dtype=int)
    nn = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            index[i, j] = nn
            nn = nn + 1
    return index


def _create_square_element_index(nx: int, ny: int):
    index = np.zeros((nx, ny), dtype=int)
    ne = 0
    for i in range(nx):
        for j in range(ny):
            index[i, j] = ne
            ne = ne + 1
    return index


def _create_space(
    shape: tuple[float, float],
    shift: tuple[float, float],
    dim: tuple[int, int],
    index: Mat[i32],
):
    nx, ny = dim
    Lx, Ly = shape
    x0, y0 = shift
    x_nodes = np.linspace(x0, Lx + x0, nx + 1)
    y_nodes = np.linspace(y0, Ly + y0, ny + 1)
    nodes = np.zeros(((nx + 1) * (ny + 1), 2), dtype=float)
    for i, x in enumerate(x_nodes):
        for j, y in enumerate(y_nodes):
            nodes[index[i, j]] = [x, y]
    return CheartMeshSpace(len(nodes), nodes)


def _create_topology(nx: int, ny: int, node_index: Mat[i32], elem_index: Mat[i32]):
    elems = np.zeros((nx * ny, 4), dtype=int)
    for i in range(nx):
        for j in range(ny):
            elems[elem_index[i, j]] = [
                node_index[i + m, j + n]
                for m, n, _ in VtkType.QuadrilateralLinear.ref_order
            ]
    return CheartMeshTopology(len(elems), elems, VtkType.QuadrilateralLinear)


def _create_boundary_side_x(
    tag: int, ix: int, iy: Vec[i32], node_index: Mat[i32], elem_index: Mat[i32]
):
    patch = np.zeros((len(iy), 2), dtype=int)
    elems = np.zeros((len(iy),), dtype=int)
    for j in iy:
        patch[j] = [node_index[ix, j + m] for m, *_ in VtkType.LineLinear.ref_order]
        elems[j] = elem_index[min(ix, len(elem_index) - 1), j]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def _create_boundary_side_y(
    tag: int, ix: Vec[i32], iy: int, node_index: Mat[i32], elem_index: Mat[i32]
):
    patch = np.zeros((len(ix), 2), dtype=int)
    elems = np.zeros((len(ix),), dtype=int)
    for i in ix:
        patch[i] = [node_index[i + m, iy] for m, *_ in VtkType.LineLinear.ref_order]
        elems[i] = elem_index[i, min(iy, len(elem_index) - 1)]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def _create_boundary_side(
    tag: int,
    ix: Vec[i32] | int,
    iy: Vec[i32] | int,
    node_index: Mat[i32],
    elem_index: Mat[i32],
):
    match ix, iy:
        case int(), np.ndarray():
            return _create_boundary_side_x(tag, ix, iy, node_index, elem_index)
        case np.ndarray(), int():
            return _create_boundary_side_y(tag, ix, iy, node_index, elem_index)
        case _:
            raise ValueError(f"Combination of {type(ix)}, {type(iy)} is not allowed")


def _create_boundary(nx: int, ny: int, node_index: Mat[i32], elem_index: Mat[i32]):
    ix = np.arange(nx, dtype=int)
    iy = np.arange(ny, dtype=int)
    bnds: dict[str | int, CheartMeshPatch] = {
        1: _create_boundary_side(1, 0, iy, node_index, elem_index),
        2: _create_boundary_side(2, nx, iy, node_index, elem_index),
        3: _create_boundary_side(3, ix, 0, node_index, elem_index),
        4: _create_boundary_side(4, ix, ny, node_index, elem_index),
    }
    return CheartMeshBoundary(len(bnds), bnds, VtkType.LineLinear)


def create_mesh(
    dim: V2[int], shape: V2[float] = (1.0, 1.0), shift: V2[float] = (0.0, 0.0)
):
    node_index = _create_square_nodal_index(*dim)
    elem_index = _create_square_element_index(*dim)
    return CheartMesh(
        _create_space(shape, shift, dim, node_index),
        _create_topology(*dim, node_index, elem_index),
        _create_boundary(*dim, node_index, elem_index),
    )
