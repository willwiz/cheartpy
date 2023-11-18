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
from cheartpy.meshing.make_grid import create_meshgrid_3D, MeshCheart
import numpy as np
import argparse

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


def wrap_nodal_positions(g: MeshCheart, r_in: float, r_out: float) -> MeshCheart:
    radius = (r_out - r_in) * np.sqrt(g.space.v[:, 0]) + r_in
    theta = 2.0 * np.pi * g.space.v[:, 1]
    g.space.v[:, 0] = radius * np.cos(theta)
    g.space.v[:, 1] = radius * np.sin(theta)
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
    g = wrap_nodal_positions(g, r_in, r_out)
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
    new_space = np.zeros((nn, g.space.v.shape[1]))
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


def create_cylinder_geometry(
    g: MeshCheart, r_in: float, r_out: float, axis: str = "z"
) -> MeshCheart:
    g = wrap_around_y(g, r_in, r_out)
    g = renormalized_mesh(g)
    g = rotate_axis(g, RotationOption[axis])
    return g


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
) -> None:
    g = create_meshgrid_3D(rn, qn, zn, 1.0, 0, 1.0, 0, length, base)
    g = create_cylinder_geometry(g, r_in, r_out, axis)
    g.write(prefix)
    print(f"!!!JOB COMPLETE!!!")


def main(args: argparse.Namespace):
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
    )


# ----  Here beging the main program  ---------------------------------------
# Get the command line arguments
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
