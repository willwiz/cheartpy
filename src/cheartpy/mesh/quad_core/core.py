from __future__ import annotations

__all__ = ["create_square_mesh"]
from typing import TYPE_CHECKING

import numpy as np

from cheartpy.cheart_mesh.data import (
    CheartMesh,
    CheartMeshBoundary,
    CheartMeshPatch,
    CheartMeshSpace,
    CheartMeshTopology,
)
from cheartpy.vtk.struct import VTKLINE1, VTKQUADRILATERAL1
from cheartpy.vtk.trait import VtkType

if TYPE_CHECKING:
    from arraystubs import T2, Arr1, Arr2


def create_square_nodal_index(nx: int, ny: int) -> Arr2[np.intc]:
    index = np.zeros((nx + 1, ny + 1), dtype=np.intc)
    nn = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            index[i, j] = nn
            nn = nn + 1
    return index


def create_square_element_index(nx: int, ny: int) -> Arr2[np.intc]:
    index = np.zeros((nx, ny), dtype=np.intc)
    ne = 0
    for i in range(nx):
        for j in range(ny):
            index[i, j] = ne
            ne = ne + 1
    return index


def create_space[I: np.integer](
    shape: tuple[float, float],
    shift: tuple[float, float],
    dim: tuple[int, int],
    index: Arr2[I],
) -> CheartMeshSpace[np.float64]:
    nx, ny = dim
    lx, ly = shape
    x0, y0 = shift
    x_nodes = np.linspace(x0, lx + x0, nx + 1)
    y_nodes = np.linspace(y0, ly + y0, ny + 1)
    nodes = np.zeros(((nx + 1) * (ny + 1), 2), dtype=np.float64)
    for i, x in enumerate(x_nodes):
        for j, y in enumerate(y_nodes):
            nodes[index[i, j]] = [x, y]
    return CheartMeshSpace(len(nodes), nodes)


def create_topology[I: np.integer](
    nx: int,
    ny: int,
    node_index: Arr2[I],
    elem_index: Arr2[I],
) -> CheartMeshTopology[I]:
    elems = np.zeros((nx * ny, 4), dtype=int)
    for i in range(nx):
        for j in range(ny):
            elems[elem_index[i, j]] = [
                node_index[i + m, j + n] for m, n, _ in VTKQUADRILATERAL1.nodes
            ]
    return CheartMeshTopology(len(elems), elems, VtkType.LinQuadrilateral)


def create_boundary_side_x_cw[I: np.integer](
    tag: int,
    ix: int,
    iy: Arr1[I],
    node_index: Arr2[I],
    elem_index: Arr2[I],
) -> CheartMeshPatch[I]:
    patch = np.zeros((len(iy), 2), dtype=int)
    elems = np.zeros((len(iy),), dtype=int)

    for j in iy:
        patch[j] = [node_index[ix + 1, j + m] for m, *_ in VTKLINE1.nodes]
        elems[j] = elem_index[ix, j]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def create_boundary_side_x_ccw[I: np.integer](
    tag: int,
    ix: int,
    iy: Arr1[I],
    node_index: Arr2[I],
    elem_index: Arr2[I],
) -> CheartMeshPatch[I]:
    patch = np.zeros((len(iy), 2), dtype=int)
    elems = np.zeros((len(iy),), dtype=int)
    for j in iy:
        patch[j] = [node_index[ix, j + m] for m, *_ in reversed(VTKLINE1.nodes)]
        elems[j] = elem_index[ix, j]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def create_boundary_side_y_cw[I: np.integer](
    tag: int,
    ix: Arr1[I],
    iy: int,
    node_index: Arr2[I],
    elem_index: Arr2[I],
) -> CheartMeshPatch[I]:
    patch = np.zeros((len(ix), 2), dtype=int)
    elems = np.zeros((len(ix),), dtype=int)
    for i in ix:
        patch[i] = [node_index[i + m, iy + 1] for m, *_ in VTKLINE1.nodes]
        elems[i] = elem_index[i, iy]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def create_boundary_side_y_ccw[I: np.integer](
    tag: int,
    ix: Arr1[I],
    iy: int,
    node_index: Arr2[I],
    elem_index: Arr2[I],
) -> CheartMeshPatch[I]:
    patch = np.zeros((len(ix), 2), dtype=int)
    elems = np.zeros((len(ix),), dtype=int)
    for i in ix:
        patch[i] = [node_index[i + m, iy] for m, *_ in reversed(VTKLINE1.nodes)]
        elems[i] = elem_index[i, iy]
    return CheartMeshPatch(tag, len(patch), elems, patch)


def create_boundary[I: np.integer](
    nx: int,
    ny: int,
    node_index: Arr2[I],
    elem_index: Arr2[I],
) -> CheartMeshBoundary[I]:
    ix = np.arange(nx, dtype=int)
    iy = np.arange(ny, dtype=int)
    bnds: dict[int, CheartMeshPatch[I]] = {
        1: create_boundary_side_x_ccw(1, 0, iy, node_index, elem_index),
        2: create_boundary_side_x_cw(2, nx - 1, iy, node_index, elem_index),
        3: create_boundary_side_y_ccw(3, ix, 0, node_index, elem_index),
        4: create_boundary_side_y_cw(4, ix, ny - 1, node_index, elem_index),
    }
    return CheartMeshBoundary(len(bnds), bnds, VtkType.LinLine)


def create_square_mesh(
    dim: T2[int],
    shape: T2[float] = (1.0, 1.0),
    shift: T2[float] = (0.0, 0.0),
) -> CheartMesh[np.float64, np.intc]:
    node_index = create_square_nodal_index(*dim)
    elem_index = create_square_element_index(*dim)
    return CheartMesh(
        create_space(shape, shift, dim, node_index),
        create_topology(*dim, node_index, elem_index),
        create_boundary(*dim, node_index, elem_index),
    )
