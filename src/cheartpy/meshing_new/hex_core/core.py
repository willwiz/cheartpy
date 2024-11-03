__all__ = ["create_mesh"]
import numpy as np
from ...var_types import *
from ...cheart_mesh import (
    CheartMesh,
    VtkType,
    CheartMeshSpace,
    CheartMeshTopology,
    CheartMeshPatch,
    CheartMeshBoundary,
)


def _create_square_nodal_index(nx: int, ny: int, nz: int):
    index = np.zeros((nx + 1, ny + 1, nz + 1), dtype=int)
    nn = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(ny + 1):
                index[i, j, k] = nn
                nn = nn + 1
    return index


def _create_square_element_index(nx: int, ny: int, nz: int):
    index = np.zeros((nx, ny), dtype=int)
    ne = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                index[i, j, k] = ne
                ne = ne + 1
    return index


def _create_space(shape: V3[float], shift: V3[float], dim: V3[int], index: MatV[i32]):
    nx, ny, nz = dim
    Lx, Ly, Lz = shape
    x0, y0, z0 = shift
    x_nodes = np.linspace(x0, Lx + x0, nx + 1)
    y_nodes = np.linspace(y0, Ly + y0, ny + 1)
    z_nodes = np.linspace(z0, Lz + z0, nz + 1)
    nodes = np.zeros(((nx + 1) * (ny + 1) * (nz + 1), 3), dtype=float)
    for i, x in enumerate(x_nodes):
        for j, y in enumerate(y_nodes):
            for k, z in enumerate(z_nodes):
                nodes[index[i, j, k]] = [x, y, z]
    return CheartMeshSpace(len(nodes), nodes)


def _create_topology(
    nx: int, ny: int, nz: int, node_index: MatV[i32], elem_index: MatV[i32]
):
    elems = np.zeros((nx * ny, 8), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                elems[elem_index[i, j, k]] = [
                    node_index[i + l, j + m, k + n]
                    for l, m, n in VtkType.HexahedronLinear.ref_order
                ]
    return CheartMeshTopology(len(elems), elems, VtkType.HexahedronLinear)


def _create_boundary_side_x(
    tag: int,
    ix: int,
    iy: Vec[i32],
    iz: Vec[i32],
    node_index: MatV[i32],
    elem_index: MatV[i32],
):
    ny, nz = len(iy), len(iz)
    patch = np.zeros((ny * nz, 4), dtype=int)
    elems = np.zeros((ny * nz,), dtype=int)
    for j in iy:
        for k in iz:
            patch[nz * j + k] = [
                node_index[ix, j + m, k + n]
                for m, n, _ in VtkType.QuadrilateralLinear.ref_order
            ]
            elems[nz * j + k] = elem_index[ix, j, k]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def _create_boundary_side_y(
    tag: int,
    ix: Vec[i32],
    iy: int,
    iz: Vec[i32],
    node_index: MatV[i32],
    elem_index: MatV[i32],
):
    nx, nz = len(ix), len(iz)
    patch = np.zeros((nx * nz, 4), dtype=int)
    elems = np.zeros((nx * nz,), dtype=int)
    for i in ix:
        for k in iz:
            patch[nz * i + k] = [
                node_index[i + m, iy, k + n]
                for m, n, _ in VtkType.QuadrilateralLinear.ref_order
            ]
            elems[nz * i + k] = elem_index[i, iy, k]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def _create_boundary_side_z(
    tag: int,
    ix: Vec[i32],
    iy: Vec[i32],
    iz: int,
    node_index: MatV[i32],
    elem_index: MatV[i32],
):
    patch = np.zeros((len(iy), 4), dtype=int)
    elems = np.zeros((len(iy),), dtype=int)
    for i in ix:
        for j in iy:
            patch[j] = [
                node_index[i + m, j + n, iz]
                for m, n, _ in VtkType.QuadrilateralLinear.ref_order
            ]
            elems[j] = elem_index[i, j, iz]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def _create_boundary_side(
    tag: int,
    ix: Vec[i32] | int,
    iy: Vec[i32] | int,
    iz: Vec[i32] | int,
    node_index: MatV[i32],
    elem_index: MatV[i32],
):
    match ix, iy, iz:
        case int(), np.ndarray(), np.ndarray():
            return _create_boundary_side_x(tag, ix, iy, iz, node_index, elem_index)
        case np.ndarray(), int(), np.ndarray():
            return _create_boundary_side_y(tag, ix, iy, iz, node_index, elem_index)
        case np.ndarray(), np.ndarray(), int():
            return _create_boundary_side_z(tag, ix, iy, iz, node_index, elem_index)
        case _:
            raise ValueError(f"Combination of {type(ix)}, {type(iy)} is not allowed")


def _create_boundary(
    nx: int, ny: int, nz: int, node_index: MatV[i32], elem_index: MatV[i32]
):
    ix = np.arange(nx + 1, dtype=int)
    iy = np.arange(ny + 1, dtype=int)
    iz = np.arange(nz + 1, dtype=int)
    bnds: dict[str | int, CheartMeshPatch] = {
        1: _create_boundary_side(1, 0, iy, iz, node_index, elem_index),
        2: _create_boundary_side(2, nx, iy, iz, node_index, elem_index),
        3: _create_boundary_side(3, ix, 0, iz, node_index, elem_index),
        4: _create_boundary_side(4, ix, ny, iz, node_index, elem_index),
        5: _create_boundary_side(5, ix, iy, 0, node_index, elem_index),
        6: _create_boundary_side(6, ix, iy, nz, node_index, elem_index),
    }
    return CheartMeshBoundary(len(bnds), bnds, VtkType.QuadrilateralLinear)


def create_mesh(
    dim: V3[int],
    shape: V3[float] = (1.0, 1.0, 1.0),
    shift: V3[float] = (0.0, 0.0, 0.0),
):
    node_index = _create_square_nodal_index(*dim)
    elem_index = _create_square_element_index(*dim)
    return CheartMesh(
        _create_space(shape, shift, dim, node_index),
        _create_topology(*dim, node_index, elem_index),
        _create_boundary(*dim, node_index, elem_index),
    )
