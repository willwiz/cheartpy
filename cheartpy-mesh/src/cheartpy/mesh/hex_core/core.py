from typing import TYPE_CHECKING

import numpy as np
from cheartpy.mesh.struct import (
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from cheartpy.vtk.struct import VTKHEXAHEDRON1, VTKQUADRILATERAL1
from cheartpy.vtk.trait import VtkType

if TYPE_CHECKING:
    from pytools.arrays import A1, A3, T3

__all__ = [
    "create_boundary",
    "create_space",
    "create_square_element_index",
    "create_square_nodal_index",
    "create_topology",
]


def create_square_nodal_index(nx: int, ny: int, nz: int) -> A3[np.intc]:
    index = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.intc)
    nn = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                index[i, j, k] = nn
                nn = nn + 1
    return index


def create_square_element_index(nx: int, ny: int, nz: int) -> A3[np.intc]:
    index = np.zeros((nx, ny, nz), dtype=np.intc)
    ne = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                index[i, j, k] = ne
                ne = ne + 1
    return index


def create_space[I: np.integer](
    shape: T3[float],
    shift: T3[float],
    dim: T3[int],
    index: A3[I],
) -> CheartMeshSpace[np.float64]:
    nx, ny, nz = dim
    lx, ly, lz = shape
    x0, y0, z0 = shift
    x_nodes = np.linspace(x0, lx + x0, nx + 1)
    y_nodes = np.linspace(y0, ly + y0, ny + 1)
    z_nodes = np.linspace(z0, lz + z0, nz + 1)
    nodes = np.zeros(((nx + 1) * (ny + 1) * (nz + 1), 3), dtype=np.float64)
    for k, z in enumerate(z_nodes):
        for j, y in enumerate(y_nodes):
            for i, x in enumerate(x_nodes):
                nodes[index[i, j, k]] = [x, y, z]
    return CheartMeshSpace(len(nodes), nodes)


def create_topology[I: np.integer](
    nx: int,
    ny: int,
    nz: int,
    node_index: A3[I],
    elem_index: A3[I],
) -> CheartMeshTopology[I]:
    elems = np.zeros((nx * ny * nz, 8), dtype=node_index.dtype)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elems[elem_index[i, j, k]] = [
                    node_index[i + p, j + m, k + n] for p, m, n in VTKHEXAHEDRON1.nodes
                ]
    return CheartMeshTopology(len(elems), elems, VtkType.LinHexahedron)


def create_boundary_side_x_cw[I: np.integer](
    tag: int,
    ip: tuple[int, A1[I], A1[I]],
    node_index: A3[I],
    elem_index: A3[I],
) -> CheartMeshPatch[I]:
    ix, iy, iz = ip
    ny, nz = len(iy), len(iz)
    patch = np.zeros((ny * nz, 4), dtype=int)
    elems = np.zeros((ny * nz,), dtype=int)
    for k in iz:
        for j in iy:
            patch[ny * k + j] = [
                node_index[ix + 1, j + m, k + n] for m, n, _ in VTKQUADRILATERAL1.nodes
            ]
            elems[ny * k + j] = elem_index[ix, j, k]
    return CheartMeshPatch(tag, len(patch), elems, patch, VtkType.LinQuadrilateral)


def create_boundary_side_x_ccw[I: np.integer](
    tag: int,
    ip: tuple[int, A1[I], A1[I]],
    node_index: A3[I],
    elem_index: A3[I],
) -> CheartMeshPatch[I]:
    ix, iy, iz = ip
    ny, nz = len(iy), len(iz)
    patch = np.zeros((ny * nz, 4), dtype=int)
    elems = np.zeros((ny * nz,), dtype=int)
    for k in iz:
        for j in iy:
            patch[ny * k + j] = [
                node_index[ix, j + m, k + n] for n, m, _ in VTKQUADRILATERAL1.nodes
            ]
            elems[ny * k + j] = elem_index[ix, j, k]
    return CheartMeshPatch(tag, len(patch), elems, patch, VtkType.LinQuadrilateral)


def create_boundary_side_y_cw[I: np.integer](
    tag: int,
    ip: tuple[A1[I], int, A1[I]],
    node_index: A3[I],
    elem_index: A3[I],
) -> CheartMeshPatch[I]:
    ix, iy, iz = ip
    nx, nz = len(ix), len(iz)
    patch = np.zeros((nx * nz, 4), dtype=int)
    elems = np.zeros((nx * nz,), dtype=int)
    for k in iz:
        for i in ix:
            patch[nx * k + i] = [
                node_index[i + m, iy + 1, k + n] for m, n, _ in VTKQUADRILATERAL1.nodes
            ]
            elems[nx * k + i] = elem_index[i, iy, k]
    return CheartMeshPatch(tag, len(patch), elems, patch, VtkType.LinQuadrilateral)


def create_boundary_side_y_ccw[I: np.integer](
    tag: int,
    ip: tuple[A1[I], int, A1[I]],
    node_index: A3[I],
    elem_index: A3[I],
) -> CheartMeshPatch[I]:
    ix, iy, iz = ip
    nx, nz = len(ix), len(iz)
    patch = np.zeros((nx * nz, 4), dtype=int)
    elems = np.zeros((nx * nz,), dtype=int)
    for k in iz:
        for i in ix:
            patch[nx * k + i] = [
                node_index[i + m, iy, k + n] for n, m, _ in VTKQUADRILATERAL1.nodes
            ]
            elems[nx * k + i] = elem_index[i, iy, k]
    return CheartMeshPatch(tag, len(patch), elems, patch, VtkType.LinQuadrilateral)


def create_boundary_side_z_cw[I: np.integer](
    tag: int,
    ip: tuple[A1[I], A1[I], int],
    node_index: A3[I],
    elem_index: A3[I],
) -> CheartMeshPatch[I]:
    ix, iy, iz = ip
    nx, ny = len(ix), len(iy)
    patch = np.zeros((nx * ny, 4), dtype=int)
    elems = np.zeros((nx * ny,), dtype=int)
    for j in iy:
        for i in ix:
            patch[nx * j + i] = [
                node_index[i + m, j + n, iz + 1] for m, n, _ in VTKQUADRILATERAL1.nodes
            ]
            elems[nx * j + i] = elem_index[i, j, iz]
    return CheartMeshPatch(tag, len(patch), elems, patch, VtkType.LinQuadrilateral)


def create_boundary_side_z_ccw[I: np.integer](
    tag: int,
    ip: tuple[A1[I], A1[I], int],
    node_index: A3[I],
    elem_index: A3[I],
) -> CheartMeshPatch[I]:
    ix, iy, iz = ip
    nx, ny = len(ix), len(iy)
    patch = np.zeros((nx * ny, 4), dtype=int)
    elems = np.zeros((nx * ny,), dtype=int)
    for j in iy:
        for i in ix:
            patch[nx * j + i] = [
                node_index[i + m, j + n, iz] for n, m, _ in VTKQUADRILATERAL1.nodes
            ]
            elems[nx * j + i] = elem_index[i, j, iz]
    return CheartMeshPatch(tag, len(patch), elems, patch, VtkType.LinQuadrilateral)


def create_boundary[I: np.integer](
    nx: int,
    ny: int,
    nz: int,
    node_index: A3[I],
    elem_index: A3[I],
) -> CheartMeshBoundary[I]:
    ix = np.arange(nx, dtype=node_index.dtype)
    iy = np.arange(ny, dtype=node_index.dtype)
    iz = np.arange(nz, dtype=node_index.dtype)
    bnds: dict[int, CheartMeshPatch[I]] = {
        1: create_boundary_side_x_ccw(1, (0, iy, iz), node_index, elem_index),
        2: create_boundary_side_x_cw(2, (nx - 1, iy, iz), node_index, elem_index),
        3: create_boundary_side_y_ccw(3, (ix, 0, iz), node_index, elem_index),
        4: create_boundary_side_y_cw(4, (ix, ny - 1, iz), node_index, elem_index),
        5: create_boundary_side_z_ccw(5, (ix, iy, 0), node_index, elem_index),
        6: create_boundary_side_z_cw(6, (ix, iy, nz - 1), node_index, elem_index),
    }
    return CheartMeshBoundary(len(bnds), bnds, VtkType.LinQuadrilateral)
