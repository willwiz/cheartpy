__all__ = [
    "create_square_nodal_index",
    "create_square_element_index",
    "create_space",
    "create_topology",
    "create_boundary_side_x",
    "create_boundary_side_y",
    "create_boundary_side_z",
    "create_boundary_side",
    "create_boundary",
]
import numpy as np
from ...var_types import *
from ...cheart_mesh import *


def create_square_nodal_index(nx: int, ny: int, nz: int):
    index = np.zeros((nx + 1, ny + 1, nz + 1), dtype=int)
    nn = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                index[i, j, k] = nn
                nn = nn + 1
    return index


def create_square_element_index(nx: int, ny: int, nz: int):
    index = np.zeros((nx, ny, nz), dtype=int)
    ne = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                index[i, j, k] = ne
                ne = ne + 1
    return index


def create_space(shape: T3[float], shift: T3[float], dim: T3[int], index: MatV[int_t]):
    nx, ny, nz = dim
    Lx, Ly, Lz = shape
    x0, y0, z0 = shift
    x_nodes = np.linspace(x0, Lx + x0, nx + 1)
    y_nodes = np.linspace(y0, Ly + y0, ny + 1)
    z_nodes = np.linspace(z0, Lz + z0, nz + 1)
    nodes = np.zeros(((nx + 1) * (ny + 1) * (nz + 1), 3), dtype=float)
    for k, z in enumerate(z_nodes):
        for j, y in enumerate(y_nodes):
            for i, x in enumerate(x_nodes):
                nodes[index[i, j, k]] = [x, y, z]
    return CheartMeshSpace(len(nodes), nodes)


def create_topology(
    nx: int, ny: int, nz: int, node_index: MatV[int_t], elem_index: MatV[int_t]
):
    elems = np.zeros((nx * ny * nz, 8), dtype=int)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elems[elem_index[i, j, k]] = [
                    node_index[i + l, j + m, k + n]
                    for l, m, n in VtkType.HexahedronLinear.ref_order
                ]
    return CheartMeshTopology(len(elems), elems, VtkType.HexahedronLinear)


def create_boundary_side_x(
    tag: int,
    ix: int,
    iy: Vec[int_t],
    iz: Vec[int_t],
    side: bool,
    node_index: MatV[int_t],
    elem_index: MatV[int_t],
):
    ny, nz = len(iy), len(iz)
    patch = np.zeros((ny * nz, 4), dtype=int)
    elems = np.zeros((ny * nz,), dtype=int)
    if side:
        for k in iz:
            for j in iy:
                patch[ny * k + j] = [
                    node_index[ix + side, j + m, k + n]
                    for m, n, _ in VtkType.QuadrilateralLinear.ref_order
                ]
                elems[ny * k + j] = elem_index[ix, j, k]
    else:
        for k in iz:
            for j in iy:
                patch[ny * k + j] = [
                    node_index[ix + side, j + m, k + n]
                    for n, m, _ in VtkType.QuadrilateralLinear.ref_order
                ]
                elems[ny * k + j] = elem_index[ix, j, k]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def create_boundary_side_y(
    tag: int,
    ix: Vec[int_t],
    iy: int,
    iz: Vec[int_t],
    side: bool,
    node_index: MatV[int_t],
    elem_index: MatV[int_t],
):
    nx, nz = len(ix), len(iz)
    patch = np.zeros((nx * nz, 4), dtype=int)
    elems = np.zeros((nx * nz,), dtype=int)
    # just swapping m and n if the side is pointing out or not
    # minimize if checks
    if side:
        for k in iz:
            for i in ix:
                patch[nx * k + i] = [
                    node_index[i + m, iy + side, k + n]
                    for m, n, _ in VtkType.QuadrilateralLinear.ref_order
                ]
                elems[nx * k + i] = elem_index[i, iy, k]
    else:
        for k in iz:
            for i in ix:
                patch[nx * k + i] = [
                    node_index[i + m, iy + side, k + n]
                    for n, m, _ in VtkType.QuadrilateralLinear.ref_order
                ]
                elems[nx * k + i] = elem_index[i, iy, k]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def create_boundary_side_z(
    tag: int,
    ix: Vec[int_t],
    iy: Vec[int_t],
    iz: int,
    side: bool,
    node_index: MatV[int_t],
    elem_index: MatV[int_t],
):
    nx, ny = len(ix), len(iy)
    patch = np.zeros((nx * ny, 4), dtype=int)
    elems = np.zeros((nx * ny,), dtype=int)
    if side:
        for j in iy:
            for i in ix:
                patch[nx * j + i] = [
                    node_index[i + m, j + n, iz + side]
                    for m, n, _ in VtkType.QuadrilateralLinear.ref_order
                ]
                elems[nx * j + i] = elem_index[i, j, iz]
    else:
        for j in iy:
            for i in ix:
                patch[nx * j + i] = [
                    node_index[i + m, j + n, iz + side]
                    for n, m, _ in VtkType.QuadrilateralLinear.ref_order
                ]
                elems[nx * j + i] = elem_index[i, j, iz]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def create_boundary_side(
    tag: int,
    ix: Vec[int_t] | int,
    iy: Vec[int_t] | int,
    iz: Vec[int_t] | int,
    side: bool,
    node_index: MatV[int_t],
    elem_index: MatV[int_t],
):
    """
    if ix is tuple, then it is the pair of (elem, node) of the side
    """
    match ix, iy, iz:
        case int(), np.ndarray(), np.ndarray():
            return create_boundary_side_x(tag, ix, iy, iz, side, node_index, elem_index)
        case np.ndarray(), int(), np.ndarray():
            return create_boundary_side_y(tag, ix, iy, iz, side, node_index, elem_index)
        case np.ndarray(), np.ndarray(), int():
            return create_boundary_side_z(tag, ix, iy, iz, side, node_index, elem_index)
        case _:
            raise ValueError(
                f"Combination of {type(ix)}, {type(iy)}, {type(iz)} is not allowed"
            )


def create_boundary(
    nx: int, ny: int, nz: int, node_index: MatV[int_t], elem_index: MatV[int_t]
):
    """
    ix, iy, iz are element list
    """
    ix = np.arange(nx, dtype=int)
    iy = np.arange(ny, dtype=int)
    iz = np.arange(nz, dtype=int)
    bnds: dict[str | int, CheartMeshPatch] = {
        1: create_boundary_side(1, 0, iy, iz, False, node_index, elem_index),
        2: create_boundary_side(2, nx - 1, iy, iz, True, node_index, elem_index),
        3: create_boundary_side(3, ix, 0, iz, False, node_index, elem_index),
        4: create_boundary_side(4, ix, ny - 1, iz, True, node_index, elem_index),
        5: create_boundary_side(5, ix, iy, 0, False, node_index, elem_index),
        6: create_boundary_side(6, ix, iy, nz - 1, True, node_index, elem_index),
    }
    return CheartMeshBoundary(len(bnds), bnds, VtkType.QuadrilateralLinear)
