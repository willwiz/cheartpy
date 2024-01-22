#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import Literal
from cheartpy.meshing.core.hexcore3D import (
    renormalized_mesh,
    mid_squish_transform_4,
    mid_squish_transform,
)
from cheartpy.meshing.core.cylindrical import (
    RotationOption,
    rotate_axis,
    create_quad_mesh_from_linear,
    cylindrical_to_cartesian,
)
from cheartpy.meshing.make_grid import (
    create_meshgrid_3D,
    MeshCheart,
)
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
parser.add_argument("--make-quad", action="store_true", help="Also make a quad mesh.")
parser.add_argument("rin", type=float, help="inner radius")
parser.add_argument("rout", type=float, help="outer radius")
parser.add_argument("length", type=float, help="longitudinal length")
parser.add_argument("base", type=float, help="base position")
parser.add_argument("rn", type=int, help="number of elements in thickness")
parser.add_argument("qn", type=int, help="number of elements in theta")
parser.add_argument("zn", type=int, help="number of elements along the central axis")


def gen_end_node_mapping(g: MeshCheart) -> dict[int, int]:
    map: dict[int, int] = dict()
    for i in range(g.zn + 1):
        for k in range(g.xn + 1):
            map[g.space.i[k, g.yn, i]] = g.space.i[k, 0, i]
    return map


def gen_cylindrical_positions(
    g: MeshCheart, r_in: float, r_out: float, length: float, base: float
) -> MeshCheart:
    g.space.v[:, 0] = (r_out - r_in) * (g.space.v[:, 0] ** 0.707) + r_in
    g.space.v[:, 1] = 2.0 * np.pi * g.space.v[:, 1]
    g.space.v[:, 2] = length * g.space.v[:, 2] + base
    return g


def wrap_around_y(
    g: MeshCheart, r_in: float, r_out: float, length: float, base: float
) -> MeshCheart:
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
    g = gen_cylindrical_positions(g, r_in, r_out, length, base)
    return g


def create_cylinder_geometry(
    g: MeshCheart, r_in: float, r_out: float, length: float, base: float
) -> MeshCheart:
    g = wrap_around_y(g, r_in, r_out, length, base)
    g = renormalized_mesh(g)
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
    make_quad: bool = False,
) -> None:
    g = create_meshgrid_3D(rn, qn, zn, 1.0, 0, 1.0, 0, 1.0, 0.0)
    g = create_cylinder_geometry(g, r_in, r_out, length, base)
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


def main_cli():
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    main_cli()
