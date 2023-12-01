import enum
from cheartpy.meshing.core.hexcore3D import (
    MeshSpace,
    MeshSurface,
    MeshTopology,
    MeshCheart,
)
import numpy as np
from numpy import ndarray as Arr

from cheartpy.meshing.core.hexcore3D import i32, f64


class RotationOption(enum.Enum):
    x = 1
    y = 2
    z = 3


def rotate_axis(g: MeshCheart, orientation: RotationOption) -> MeshCheart:
    if orientation is RotationOption.z:
        return g
    elif orientation is RotationOption.x:
        mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ np.array(
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        )
    elif orientation is RotationOption.y:
        mat = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]) @ np.array(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        )
    g.space.v = g.space.v @ mat.T
    return g


def generate_new_topology_nodes(top: Arr[int, i32]) -> list[frozenset[int]]:
    return [
        frozenset([top[0], top[1]]),
        frozenset([top[0], top[2]]),
        frozenset([top[0], top[1], top[2], top[3]]),
        frozenset([top[1], top[3]]),
        frozenset([top[2], top[3]]),
        frozenset([top[0], top[4]]),
        frozenset([top[0], top[1], top[4], top[5]]),
        frozenset([top[1], top[5]]),
        frozenset([top[0], top[2], top[4], top[6]]),
        frozenset(top),
        frozenset([top[1], top[3], top[5], top[7]]),
        frozenset([top[2], top[6]]),
        frozenset([top[2], top[3], top[6], top[7]]),
        frozenset([top[3], top[7]]),
        frozenset([top[4], top[5]]),
        frozenset([top[4], top[6]]),
        frozenset([top[4], top[5], top[6], top[7]]),
        frozenset([top[5], top[7]]),
        frozenset([top[6], top[7]]),
    ]


def generate_new_surface_nodes(surf: Arr[int, i32]) -> list[frozenset[int]]:
    return [
        frozenset([surf[0], surf[1]]),
        frozenset([surf[0], surf[2]]),
        frozenset([surf[0], surf[1], surf[2], surf[3]]),
        frozenset([surf[1], surf[3]]),
        frozenset([surf[2], surf[3]]),
    ]


def get_mid_point(
    nodes: Arr[tuple[int, int], f64], node_pts: frozenset[int]
) -> Arr[int, f64]:
    node_pos = nodes[list(node_pts)]
    theta = np.remainder(node_pos[:, 1] - np.min(node_pos[:, 1]), 2 * np.pi) + np.min(
        node_pos[:, 1]
    )
    r = np.mean(node_pos[:, 0])
    q = np.remainder(
        np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))), 2 * np.pi
    )
    z = np.mean(node_pos[:, 2])
    # print(f"{np.isclose(q, np.mean(theta))}, {np.mean(theta)}, {q=}")
    upper_side = theta > q
    if sum(upper_side) == len(node_pts) or sum(upper_side) == 0:
        return np.array([r, q, z], dtype=float)
    dq = np.mean(theta[upper_side]) - np.mean(theta[upper_side == False])
    wrapping = dq < np.pi
    dq = wrapping * dq + (not wrapping) * (2 * np.pi - dq)
    # print(f"{dq=}")
    return np.array(
        [r * (3.0 + np.cos(dq)) / (4.0 * np.cos(0.5 * dq)), q, z], dtype=float
    )


def cylindrical_to_cartesian(g: MeshCheart) -> MeshCheart:
    radius = g.space.v[:, 0]
    theta = g.space.v[:, 1]
    g.space.v[:, 0], g.space.v[:, 1] = radius * np.cos(theta), radius * np.sin(theta)
    return g


def init_quad_mesh_for_cyclic(g: MeshCheart) -> MeshCheart:
    g_quad = MeshCheart(g.xn, g.yn, g.zn, order=2)
    for name, b in g.surfs.items():
        g_quad.surfs[name] = MeshSurface(b.n, b.tag, order=2)
        g_quad.surfs[name].key = b.key
    return g_quad


def create_node_map(g: MeshCheart) -> dict[frozenset[int], int]:
    nn = g.space.n
    node_map = {frozenset([i]): i for i in range(nn)}
    quad_top_nodes = [generate_new_topology_nodes(elem) for elem in g.top.v]
    for elem in quad_top_nodes:
        for pt in elem:
            if pt not in node_map:
                node_map[pt] = nn
                nn = nn + 1
    return node_map


def create_quad_space(node_map: dict[frozenset[int], int], x: MeshSpace):
    nn = len(node_map)
    new_nodes = np.zeros((nn, 3), dtype=float)
    for k, v in node_map.items():
        if len(k) == 1:
            [m] = k
            new_nodes[v] = x.v[m]
        elif len(k) > 1:
            new_nodes[v] = get_mid_point(x.v, k)
        else:
            raise ValueError(f"Empty set found for generating new nodes")
    return nn, new_nodes


def create_quad_topology(node_map: dict[frozenset[int], int], t: MeshTopology):
    new_top = np.zeros((t.n, 27), dtype=int)
    for i, elem in enumerate(t.v):
        new_top[i, :8] = elem
        new_nodes = generate_new_topology_nodes(elem)
        for j, v in enumerate(new_nodes, start=8):
            new_top[i, j] = node_map[v]
    return new_top


def create_quad_surface(node_map: dict[frozenset[int], int], b: MeshSurface):
    new_surf = np.zeros((b.n, 9), dtype=int)
    for i, patch in enumerate(b.v):
        new_surf[i, :4] = patch
        new_nodes = generate_new_surface_nodes(patch)
        for j, v in enumerate(new_nodes, start=4):
            new_surf[i, j] = node_map[v]
    return new_surf


def create_quad_mesh_from_linear(g: MeshCheart):
    g_quad = init_quad_mesh_for_cyclic(g)
    node_map = create_node_map(g)
    g_quad.space.n, g_quad.space.v = create_quad_space(node_map, g.space)
    g_quad.top.v[:] = create_quad_topology(node_map, g.top)
    for k, v in g.surfs.items():
        g_quad.surfs[k].v[:] = create_quad_surface(node_map, v)
    return g_quad
