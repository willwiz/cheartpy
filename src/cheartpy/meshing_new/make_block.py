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
from .hex_core import create_hex_mesh
import argparse

parser = argparse.ArgumentParser("block", description="Make a cube")
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


def main(args: argparse.Namespace):
    mesh = create_hex_mesh(
        (args.xn, args.yn, args.zn),
        (args.xsize, args.ysize, args.zsize),
        (args.offset[0], args.offset[1], args.offset[2]),
    )
    mesh.save(args.prefix)


# ----  Here beging the main program  ---------------------------------------
# Get the command line arguments
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
