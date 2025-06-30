__all__ = ["compute_bc_w", "create_noise", "update_disp_w_noise"]
from collections import defaultdict
from typing import cast

import numpy as np
from arraystubs import Arr, Arr1, Arr2
from cheartpy.io.api import chread_d, chwrite_d_utf
from cheartpy.mesh.struct import CheartMesh
from numpy import interp


def unbias[T: np.floating, D: (tuple[int], tuple[int, int], tuple[int, int, int])](
    vals: Arr[D, T],
) -> Arr[D, T]:
    return cast("Arr[D, T]", vals - np.mean(vals))


def generate_noise_data(
    mag: float,
    nx: int,
    ny: int,
) -> tuple[tuple[Arr1[np.float64], Arr1[np.float64]], Arr2[np.float64]]:
    noise = np.random.default_rng().normal(0.0, mag, (nx, ny))
    dx = 0.5 / nx
    eps = np.pad(noise, ((1, 1), (0, 0)), mode="wrap")
    eps = np.pad(eps, ((0, 0), (1, 1)), mode="constant", constant_values=0)
    x = np.linspace(-dx, 1 + dx, nx + 2)
    y = np.linspace(0, 1, ny + 2)
    return (x, y), eps


def create_noise[F: np.floating](
    mag: float,
    cl: Arr2[F],
    normal: Arr2[F],
    spatial_freq: tuple[int, int] = (3, 5),
    bc_w: Arr1[F] | None = None,
) -> Arr2[F]:
    (x, xp), yp = generate_noise_data(mag, *spatial_freq)

    y = interp(x, xp, yp, cl, method="cubic")
    noise = unbias(y)
    if bc_w is not None:
        noise = bc_w * noise
    return noise[:, None] * normal


def update_disp_w_noise[F: np.floating](
    prefix: str,
    cl: Arr2[F],
    normal: Arr2[F],
    mag: float,
    bc_w: Arr1[F] | None = None,
) -> None:
    noise = create_noise(mag, cl, normal, bc_w=bc_w)
    disp = chread_d(f"{prefix}") + noise
    chwrite_d_utf(f"{prefix}", disp)
    chwrite_d_utf(f"{prefix}", disp)


def find_neighbours[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> dict[int, set[int]]:
    neighbours: dict[int, set[int]] = defaultdict(set)
    for elem in mesh.top.v:
        for node in elem:
            neighbours[int(node)].update({int(i) for i in elem} - {int(node)})
    return neighbours


def compute_bc_w[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    surfs: list[int],
    mult: float = 0.5,
    nest: int = 3,
) -> Arr1[F]:
    if mesh.bnd is None:
        msg = "No boundary vertices found"
        raise ValueError(msg)
    bc_w = np.zeros(mesh.space.n, dtype=mesh.space.v.dtype)
    bc_nodes: Arr1[I] = np.unique(
        [mesh.bnd.v[i].v for i in surfs],
    )
    bc_w[bc_nodes] = 1.0
    neighbors = find_neighbours(mesh)
    current: set[int] = {int(i) for i in bc_nodes}
    neighbour_list = [neighbors[n] for n in current]
    all_neighbours: set[int] = set()
    new_nodes: set[int] = set()
    all_neighbours = all_neighbours.union(*neighbour_list)
    new_nodes: set[int] = all_neighbours - current
    for _ in range(nest):
        for k in new_nodes:
            nw = list(neighbors[k])
            bc_w[k] = mult * np.amax(bc_w[nw])
        current = current | new_nodes
        new_nodes = new_nodes.union(*[neighbors[n] for n in current]) - current
    bc_w[bc_w > 1] = 1
    return (1 - bc_w).astype(bc_w.dtype)


def diffuse_bc_w[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
    surfs: list[int],
    mult: float = 1.0,
    nest: int = 20,
) -> Arr1[F]:
    if mesh.bnd is None:
        msg = "No boundary vertices found"
        raise ValueError(msg)
    bc_w: Arr1[F] = np.zeros(mesh.space.n, dtype=mesh.space.v.dtype)
    bc_nodes: Arr1[I] = np.unique(
        [n for i in surfs for n in mesh.bnd.v[i].v.flatten()],
    )
    bc_w[bc_nodes] = 1.0
    neighbors = find_neighbours(mesh)
    current_neighbours: set[int] = {int(i) for i in bc_nodes}
    current: set[int] = set()
    for _ in range(nest):
        current = current.union(*[neighbors[n] for n in current_neighbours])
        snap_shot = bc_w.copy()
        for k in current_neighbours:
            nw = list(neighbors[k])
            bc_w[k] = mult * snap_shot[nw].mean()
        bc_w[bc_nodes] = 1.0
    return (1.0 - bc_w).astype(bc_w.dtype)
