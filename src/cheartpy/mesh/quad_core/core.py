__all__ = ["create_square_mesh"]
import numpy as np
from ...var_types import *
from ...cheart_mesh import *


def create_square_nodal_index(nx: int, ny: int):
    index = np.zeros((nx + 1, ny + 1), dtype=int)
    nn = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            index[i, j] = nn
            nn = nn + 1
    return index


def create_square_element_index(nx: int, ny: int):
    index = np.zeros((nx, ny), dtype=int)
    ne = 0
    for i in range(nx):
        for j in range(ny):
            index[i, j] = ne
            ne = ne + 1
    return index


def create_space(
    shape: tuple[float, float],
    shift: tuple[float, float],
    dim: tuple[int, int],
    index: Mat[int_t],
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


def create_topology(nx: int, ny: int, node_index: Mat[int_t], elem_index: Mat[int_t]):
    elems = np.zeros((nx * ny, 4), dtype=int)
    for i in range(nx):
        for j in range(ny):
            elems[elem_index[i, j]] = [
                node_index[i + m, j + n]
                for m, n, _ in VtkType.QuadrilateralLinear.ref_order
            ]
    return CheartMeshTopology(len(elems), elems, VtkType.QuadrilateralLinear)


def create_boundary_side_x(
    tag: int,
    ix: int,
    iy: Vec[int_t],
    side: bool,
    node_index: Mat[int_t],
    elem_index: Mat[int_t],
):
    patch = np.zeros((len(iy), 2), dtype=int)
    elems = np.zeros((len(iy),), dtype=int)
    if side:
        for j in iy:
            patch[j] = [
                node_index[ix + side, j + m] for m, *_ in VtkType.LineLinear.ref_order
            ]
            elems[j] = elem_index[ix, j]
    else:
        for j in iy:
            patch[j] = [
                node_index[ix + side, j + m]
                for m, *_ in reversed(VtkType.LineLinear.ref_order)
            ]
            elems[j] = elem_index[ix, j]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def create_boundary_side_y(
    tag: int,
    ix: Vec[int_t],
    iy: int,
    side: bool,
    node_index: Mat[int_t],
    elem_index: Mat[int_t],
):
    patch = np.zeros((len(ix), 2), dtype=int)
    elems = np.zeros((len(ix),), dtype=int)
    if side:
        for i in ix:
            patch[i] = [
                node_index[i + m, iy + side] for m, *_ in VtkType.LineLinear.ref_order
            ]
            elems[i] = elem_index[i, iy]
    else:
        for i in ix:
            patch[i] = [
                node_index[i + m, iy + side]
                for m, *_ in reversed(VtkType.LineLinear.ref_order)
            ]
            elems[i] = elem_index[i, iy]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def create_boundary_side(
    tag: int,
    ix: Vec[int_t] | int,
    iy: Vec[int_t] | int,
    side: bool,
    node_index: Mat[int_t],
    elem_index: Mat[int_t],
):
    match ix, iy:
        case int(), np.ndarray():
            return create_boundary_side_x(tag, ix, iy, side, node_index, elem_index)
        case np.ndarray(), int():
            return create_boundary_side_y(tag, ix, iy, side, node_index, elem_index)
        case _:
            raise ValueError(f"Combination of {type(ix)}, {type(iy)} is not allowed")


def _create_boundary(nx: int, ny: int, node_index: Mat[int_t], elem_index: Mat[int_t]):
    ix = np.arange(nx, dtype=int)
    iy = np.arange(ny, dtype=int)
    bnds: dict[str | int, CheartMeshPatch] = {
        1: create_boundary_side(1, 0, iy, False, node_index, elem_index),
        2: create_boundary_side(2, nx - 1, iy, True, node_index, elem_index),
        3: create_boundary_side(3, ix, 0, False, node_index, elem_index),
        4: create_boundary_side(4, ix, ny - 1, True, node_index, elem_index),
    }
    return CheartMeshBoundary(len(bnds), bnds, VtkType.LineLinear)


def create_square_mesh(
    dim: V2[int], shape: V2[float] = (1.0, 1.0), shift: V2[float] = (0.0, 0.0)
):
    node_index = create_square_nodal_index(*dim)
    elem_index = create_square_element_index(*dim)
    return CheartMesh(
        create_space(shape, shift, dim, node_index),
        create_topology(*dim, node_index, elem_index),
        _create_boundary(*dim, node_index, elem_index),
    )
