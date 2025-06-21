__all__ = ["compute_bc_w", "create_noise", "update_disp_w_noise"]
from collections import defaultdict
from typing import cast

import numpy as np

# from scipy.interpolate import interpn  # type: ignore
from numpy import interp as interpn

from ..cheart_mesh.data import CheartMesh
from ..cheart_mesh.io import chread_d, chwrite_d_utf
from ..var_types import *


def unbias[T: (Vec[f64], Mat[f64], MatV[f64])](vals: T) -> T:
    return vals - np.mean(vals)


def generate_noise_data(
    mag: float,
    nx: int,
    ny: int,
) -> tuple[tuple[Vec[f64], Vec[f64]], Mat[f64]]:
    noise = np.random.normal(0.0, mag, (nx, ny))
    dx = 0.5 / nx
    eps = np.pad(noise, ((1, 1), (0, 0)), mode="wrap")
    eps = np.pad(eps, ((0, 0), (1, 1)), mode="constant", constant_values=0)
    x = np.linspace(-dx, 1 + dx, nx + 2)
    y = np.linspace(0, 1, ny + 2)
    return (x, y), eps


def create_noise(
    mag: float,
    cl: Mat[f64],
    normal: Mat[f64],
    spatial_freq: tuple[int, int] = (3, 5),
    bc_w: Vec[f64] | None = None,
) -> Mat[f64]:
    noise_data = generate_noise_data(mag, *spatial_freq)
    noise = unbias(cast("Vec[f64]", interpn(*noise_data, cl, method="cubic")))
    if bc_w is not None:
        noise = bc_w * noise
    return noise[:, None] * normal


def update_disp_w_noise(
    prefix: str,
    cl: Mat[f64],
    normal: Mat[f64],
    mag: float,
    bc_w: Vec[f64] | None = None,
):
    noise = create_noise(mag, cl, normal, bc_w=bc_w)
    disp = chread_d(f"{prefix}") + noise
    chwrite_d_utf(f"{prefix}", disp)
    chwrite_d_utf(f"{prefix}", disp)


def find_neighbours(mesh: CheartMesh):
    neighbours: dict[int, set[int]] = defaultdict(set)
    for elem in mesh.top.v:
        for node in elem:
            neighbours[node].update(set(elem) - {node})
    return neighbours


def compute_bc_w(
    mesh: CheartMesh,
    surfs: list[int],
    mult: float = 0.5,
    nest: int = 3,
) -> Vec[f64]:
    if mesh.bnd is None:
        raise ValueError("No boundary vertices found")
    bc_w: Vec[f64] = np.zeros(mesh.space.n, dtype=float)
    bc_nodes: Vec[int_t] = np.unique(
        [n for i in surfs for n in mesh.bnd.v[i].v.flatten()],
    )
    bc_w[bc_nodes] = 1.0
    neighbors = find_neighbours(mesh)
    current: set[int] = set(bc_nodes)
    new_nodes: set[int] = set().union(*[neighbors[n] for n in current]) - current
    for _ in range(nest):
        for k in new_nodes:
            nw = list(neighbors[k])
            bc_w[k] = mult * np.amax(bc_w[nw])
        current = current | new_nodes
        new_nodes = set().union(*[neighbors[n] for n in current]) - current
    # bc_w[bc_w > 1] = 1
    return 1 - bc_w


def diffuse_bc_w(mesh: CheartMesh, surfs: list[int], mult: float = 1.0, nest: int = 20):
    if mesh.bnd is None:
        raise ValueError("No boundary vertices found")
    bc_w: Vec[f64] = np.zeros(mesh.space.n, dtype=float)
    bc_nodes: Vec[int_t] = np.unique(
        [n for i in surfs for n in mesh.bnd.v[i].v.flatten()],
    )
    bc_w[bc_nodes] = 1.0
    neighbors = find_neighbours(mesh)
    current: set[int] = set(bc_nodes)
    for _ in range(nest):
        current: set[int] = set().union(*[neighbors[n] for n in current])
        snap_shot = bc_w.copy()
        for k in current:
            nw = list(neighbors[k])
            bc_w[k] = mult * snap_shot[nw].mean()
        bc_w[bc_nodes] = 1.0
    return 1.0 - bc_w
