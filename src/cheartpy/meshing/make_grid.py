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

import typing as tp
import numpy as np
from numpy.typing import NDArray as Arr
from numpy import float64 as dbl
from numpy import int32 as spi
import argparse

parser = argparse.ArgumentParser("mesh", description="Make a cube")
parser.add_argument(
    "--prefix", "-p", type=str, default="cube", help="Prefix for saved file."
)
parser.add_argument(
    "--offset",
    type=float,
    nargs=3,
    default=[0, 0, 0],
    metavar=("xoff", "yoff", "zoff"),
    help="starting corner should be shifted by",
)
parser.add_argument("xsize", type=float, help="x length")
parser.add_argument("xn", type=int, help="number of elements in x")
parser.add_argument("ysize", type=float, help="y length")
parser.add_argument("yn", type=int, help="number of elements in y")
parser.add_argument("zsize", type=float, help="z length")
parser.add_argument("zn", type=int, help="number of elements in z")


# This is gives the order of the nodes in an elent given a starting indice
vorder = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]


class Mesh:
    n: int
    i: Arr[spi]
    v: Arr[dbl]

    def __getitem__(self, key):
        # if key is of invalid type or value, the list values will raise the error
        return self.v[self.i[key]]


class MeshSpace:
    __slots__ = ["n", "i", "v"]
    n: int
    i: Arr[spi]
    v: Arr[dbl]

    def __init__(self, xn: int, yn: int, zn: int, dim: int = 3) -> None:
        self.n = (xn + 1) * (yn + 1) * (zn + 1)
        self.i = np.zeros((xn + 1, yn + 1, zn + 1), dtype=int)
        self.v = np.zeros((self.n, dim))

    def write(self, f: tp.TextIO) -> None:
        for v in self.v:
            for x in v:
                f.write(f"{x:>22.16}")
            f.write("\n")


class MeshTopology:
    __slots__ = ["n", "i", "v"]
    n: int
    i: Arr[spi]
    v: Arr[spi]

    def __init__(self, xn: int, yn: int, zn: int, dim: int = 3, order: int = 1) -> None:
        self.n = xn * yn * zn
        self.i = np.zeros((xn, yn, zn), dtype=int)
        self.v = np.zeros((self.n, (order + 1) ** dim), dtype=int)

    def write(self, f: tp.TextIO) -> None:
        for elem in self.v:
            for node in elem:
                f.write(f"{node + 1:>12d}")
            f.write("\n")


class MeshSurface:
    __slots__ = ["n", "tag", "key", "v"]
    n: int
    tag: int
    key: Arr[spi]
    v: Arr[spi]

    def __init__(self, npatch: int, tag: int, dim: int = 3, order: int = 1) -> None:
        self.n = npatch
        self.v = np.zeros((npatch, (order + 1) ** (dim - 1)), dtype=int)
        self.key = np.zeros((npatch,), dtype=int)
        self.tag = tag

    def write(self, f: tp.TextIO):
        for k, v in zip(self.key, self.v):
            f.write(f"{k + 1:>12d}")
            for x in v:
                f.write(f"{x + 1:>12d}")
            f.write(f"{self.tag:>12d}\n")


class MeshCheart:
    space: MeshSpace
    top: MeshTopology
    surfs: dict[str, MeshSurface]

    def __init__(self, xn: int, yn: int, zn: int, dim: int = 3, order: int = 1) -> None:
        self.xn = xn
        self.yn = yn
        self.zn = zn
        self.space = MeshSpace(xn, yn, zn, dim=dim)
        self.top = MeshTopology(xn, yn, zn, dim=dim, order=order)
        self.surfs = dict()

    def write(self, prefix: str):
        with open(prefix + "_FE.X", "w") as f:
            print(f"Writing space to {f.name}")
            f.write(f"{self.space.n:>12d}{3:>12d}\n")
            self.space.write(f=f)

        with open(prefix + "_FE.T", "w") as f:
            print(f"Writing topology to {f.name}")
            f.write(f"{self.top.n:>12d}{self.space.n:>12d}\n")
            self.top.write(f=f)

        with open(prefix + "_FE.B", "w") as f:
            print(f"Writing boundary to {f.name}")
            npatch = sum([v.n for v in self.surfs.values()])
            f.write(f"{npatch:>12d}\n")
            for _, surf in self.surfs.items():
                surf.write(f=f)


def compute_space(g: MeshCheart, xsize, xoff, ysize, yoff, zsize, zoff) -> None:
    dx = xsize / g.xn
    dy = ysize / g.yn
    dz = zsize / g.zn
    # fill array
    m = 0
    for i in range(g.zn + 1):
        for j in range(g.yn + 1):
            for k in range(g.xn + 1):
                g.space.v[m] = [k * dx + xoff, j * dy + yoff, i * dz + zoff]
                g.space.i[k, j, i] = m
                m = m + 1


def compute_topology(g: MeshCheart) -> None:
    l = 0
    for i in range(g.zn):
        for j in range(g.yn):
            for k in range(g.xn):
                for m in range(8):
                    g.top.v[i * g.yn * g.xn + j * g.xn + k, m] = g.space.i[
                        k + vorder[m][0], j + vorder[m][1], i + vorder[m][2]
                    ]
                g.top.i[k, j, i] = l
                l = l + 1


def compute_surface(g: MeshCheart) -> None:
    # for the -x surface
    s = MeshSurface(g.yn * g.zn, 1)
    for i in range(g.zn):
        for j in range(g.yn):
            k = i * g.yn + j
            s.key[k] = g.top.i[0, j, i]
            s.v[k, 0] = g.space.i[0, j, i]
            s.v[k, 1] = g.space.i[0, j + 1, i]
            s.v[k, 2] = g.space.i[0, j, i + 1]
            s.v[k, 3] = g.space.i[0, j + 1, i + 1]
    g.surfs["-x"] = s
    # for the +x surface
    s = MeshSurface(g.yn * g.zn, 2)
    for i in range(g.zn):
        for j in range(g.yn):
            k = i * g.yn + j
            s.key[k] = g.top.i[g.xn - 1, j, i]
            s.v[k, 0] = g.space.i[g.xn, j, i]
            s.v[k, 1] = g.space.i[g.xn, j + 1, i]
            s.v[k, 2] = g.space.i[g.xn, j, i + 1]
            s.v[k, 3] = g.space.i[g.xn, j + 1, i + 1]
    g.surfs["+x"] = s
    # for the -y surface
    s = MeshSurface(g.xn * g.zn, 3)
    for i in range(g.zn):
        for k in range(g.xn):
            j = i * g.xn + k
            s.key[j] = g.top.i[k, 0, i]
            s.v[j, 0] = g.space.i[k, 0, i]
            s.v[j, 1] = g.space.i[k + 1, 0, i]
            s.v[j, 2] = g.space.i[k, 0, i + 1]
            s.v[j, 3] = g.space.i[k + 1, 0, i + 1]
    g.surfs["-y"] = s
    # for the +y surface
    s = MeshSurface(g.xn * g.zn, 4)
    for i in range(g.zn):
        for k in range(g.xn):
            j = i * g.xn + k
            s.key[j] = g.top.i[k, g.yn - 1, i]
            s.v[j, 0] = g.space.i[k, g.yn, i]
            s.v[j, 1] = g.space.i[k + 1, g.yn, i]
            s.v[j, 2] = g.space.i[k, g.yn, i + 1]
            s.v[j, 3] = g.space.i[k + 1, g.yn, i + 1]
    g.surfs["+y"] = s
    # for the -z surface
    s = MeshSurface(g.xn * g.yn, 5)
    for j in range(g.yn):
        for k in range(g.xn):
            i = j * g.xn + k
            s.key[i] = g.top.i[k, j, 0]
            s.v[i, 0] = g.space.i[k, j, 0]
            s.v[i, 1] = g.space.i[k + 1, j, 0]
            s.v[i, 2] = g.space.i[k, j + 1, 0]
            s.v[i, 3] = g.space.i[k + 1, j + 1, 0]
    g.surfs["-z"] = s
    # for the +z surface
    s = MeshSurface(g.xn * g.yn, 6)
    for j in range(g.yn):
        for k in range(g.xn):
            i = j * g.xn + k
            s.key[i] = g.top.i[k, j, g.zn - 1]
            s.v[i, 0] = g.space.i[k, j, g.zn]
            s.v[i, 1] = g.space.i[k + 1, j, g.zn]
            s.v[i, 2] = g.space.i[k, j + 1, g.zn]
            s.v[i, 3] = g.space.i[k + 1, j + 1, g.zn]
    g.surfs["+z"] = s


def create_meshgrid_3D(
    xn: int,
    yn: int,
    zn: int,
    xsize: float,
    xoff: float,
    ysize: float,
    yoff: float,
    zsize: float,
    zoff: float,
) -> MeshCheart:
    g = MeshCheart(xn, yn, zn)
    compute_space(g, xsize, xoff, ysize, yoff, zsize, zoff)
    compute_topology(g)
    compute_surface(g)
    return g


def create_cheart_mesh(
    xn, yn, zn, xsize, xoff, ysize, yoff, zsize, zoff, prefix: str
) -> None:
    g = create_meshgrid_3D(xn, yn, zn, xsize, xoff, ysize, yoff, zsize, zoff)
    g.write(prefix)
    print(f"!!!JOB COMPLETE!!!")


def main(args: argparse.Namespace):
    create_cheart_mesh(
        args.xn,
        args.yn,
        args.zn,
        args.xsize,
        args.offset[0],
        args.ysize,
        args.offset[1],
        args.zsize,
        args.offset[2],
        args.prefix,
    )


# ----  Here beging the main program  ---------------------------------------
# Get the command line arguments
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
