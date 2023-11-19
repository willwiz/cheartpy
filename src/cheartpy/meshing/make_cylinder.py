#!/usr/bin/python3
# -*- coding: utf-8 -*-

# This creates an hexahedral mesh of uniform grids
# The inputs of this script are:
#     x_length x_n x_offset y_length y_n y_offset z_length z_n z_offset fileout
# This 3 files created are:
#     Topology.T
#        n dim
#        node1 node2 ...
#        .
#        .
#
#     Node.X
#        n dim
#        x1 x2 x3 ...
#        .
#        .
#
#     Boundary.B
#        n
#        elem node1 node2 ... patch#
#        .
#        .
#

import enum
from typing import Literal
from cheartpy.meshing.make_grid import (
    MeshSpace,
    MeshSurface,
    MeshTopology,
    create_meshgrid_3D,
    MeshCheart,
)
import numpy as np
from numpy import ndarray as Arr
import argparse

f64 = np.dtype[np.float64]
i32 = np.dtype[np.int32]

parser = argparse.ArgumentParser("mesh", description="Make a cube")
parser.add_argument(
    "--prefix", "-p", type=str, default="cube", help="Prefix for saved file."
)
parser.add_argument(
    "--axis",
    "-a",
    type=str,
    default="z",
    choices={"x", "y", "z"},
    help="Which cartesian axis should the central axis be in.",
)
parser.add_argument("--make-quad", action="store_true", help="Also make a quad mesh.")
parser.add_argument("rin", type=float, help="inner radius")
parser.add_argument("rout", type=float, help="outer radius")
parser.add_argument("length", type=float, help="longitudinal length")
parser.add_argument("base", type=float, help="base position")
parser.add_argument("rn", type=int, help="number of elements in thickness")
parser.add_argument("qn", type=int, help="number of elements in theta")
parser.add_argument("zn", type=int, help="number of elements in z")


def gen_end_node_mapping(g: MeshCheart) -> dict[int, int]:
    map: dict[int, int] = dict()
    for i in range(g.zn + 1):
        for k in range(g.xn + 1):
            map[g.space.i[k, g.yn, i]] = g.space.i[k, 0, i]
    return map


def gen_cylindrical_positions(g: MeshCheart, r_in: float, r_out: float) -> MeshCheart:
    g.space.v[:, 0] = (r_out - r_in) * np.sqrt(g.space.v[:, 0]) + r_in
    g.space.v[:, 1] = 2.0 * np.pi * g.space.v[:, 1]
    return g


def wrap_around_y(g: MeshCheart, r_in: float, r_out: float) -> MeshCheart:
    map = gen_end_node_mapping(g)
    for i, row in enumerate(g.top.v):
        for j, v in enumerate(row):
            if v in map:
                g.top.v[i, j] = map[v]
            else:
                g.top.v[i, j] = v
    del g.surfs["-y"]
    del g.surfs["+y"]
    g.surfs["-z"].tag = 1
    g.surfs["+z"].tag = 2
    g.surfs["-x"].tag = 3
    g.surfs["+x"].tag = 4
    for b in g.surfs.values():
        for i, row in enumerate(b.v):
            for j, v in enumerate(row):
                if v in map:
                    b.v[i, j] = map[v]
                else:
                    b.v[i, j] = v
    g = gen_cylindrical_positions(g, r_in, r_out)
    return g


def build_elmap(g: MeshCheart) -> tuple[dict[int, int], int]:
    uniques = np.unique(g.top.v)
    elmap: dict[int, int] = dict()
    nn = 0
    for p in uniques:
        elmap[p] = nn
        nn = nn + 1
    return elmap, nn


def renormalized_mesh(g: MeshCheart) -> MeshCheart:
    elmap, nn = build_elmap(g)
    new_space = np.zeros((nn, g.space.v.shape[1]), dtype=float)
    for k, v in elmap.items():
        new_space[v] = g.space.v[k]
    g.space.n = nn
    g.space.v = new_space
    for i, row in enumerate(g.top.v):
        for j, v in enumerate(row):
            g.top.v[i, j] = elmap[v]
    for b in g.surfs.values():
        for i, row in enumerate(b.v):
            for j, v in enumerate(row):
                b.v[i, j] = elmap[v]
    return g


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


def create_cylinder_geometry(g: MeshCheart, r_in: float, r_out: float) -> MeshCheart:
    g = wrap_around_y(g, r_in, r_out)
    g = renormalized_mesh(g)
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
    r = np.mean(node_pos[:, 0])
    q = np.mean(node_pos[:, 1])
    z = np.mean(node_pos[:, 2])
    theta = node_pos[:, 1]
    upper_side = theta > q
    if sum(upper_side) == len(node_pts) or sum(upper_side) == 0:
        return np.array([r, q, z], dtype=float)
    dq = np.mean(theta[upper_side]) - np.mean(theta[upper_side == False])
    return np.array(
        [r * (3.0 + np.cos(0.5 * dq)) / (4.0 * np.cos(0.5 * dq)), q, z], dtype=float
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


def create_cheart_mesh(
    prefix: str,
    r_in: float,
    r_out: float,
    length: float,
    base: float,
    rn: int,
    qn: int,
    zn: int,
    axis: Literal["x", "y", "z"] = "z",
    make_quad: bool = False,
) -> None:
    g = create_meshgrid_3D(rn, qn, zn, 1.0, 0, 1.0, 0, length, base)
    g = create_cylinder_geometry(g, r_in, r_out)
    if make_quad:
        g_quad = create_quad_mesh_from_linear(g)
        g_quad = cylindrical_to_cartesian(g_quad)
        g_quad = rotate_axis(g_quad, RotationOption[axis])
        g_quad.write(f"{prefix}_quad")
    g = cylindrical_to_cartesian(g)
    g = rotate_axis(g, RotationOption[axis])
    g.write(prefix)
    print(f"!!!JOB COMPLETE!!!")


# ----  Here beging the main program  ---------------------------------------
# Get the command line arguments
def main(args: argparse.Namespace):
    print(args)
    if args.qn < 3:
        raise ValueError(f"Number of circumferential elements must be greater than 2")
    create_cheart_mesh(
        args.prefix,
        args.rin,
        args.rout,
        args.length,
        args.base,
        args.rn,
        args.qn,
        args.zn,
        args.axis,
        args.make_quad,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
