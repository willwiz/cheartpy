#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import Literal, cast
from ..var_types import *
from .cylinder_core.api import create_cylinder_mesh
import argparse

parser = argparse.ArgumentParser("block", description="Make a cube")
parser.add_argument(
    "--prefix", "-p", type=str, default="cube", help="Prefix for saved file."
)
parser.add_argument("-l", type=float, default=1, help="long axis length")
parser.add_argument("-b", type=float, default=0, help="starting location")
parser.add_argument(
    "--axis",
    "-a",
    type=str,
    default="z",
    choices={"x", "y", "z"},
    help="Which cartesian axis should the central axis be in.",
)
parser.add_argument("--make-quad", action="store_true", help="auto make a quad mesh")
parser.add_argument("rin", type=int, help="number of elements in r")
parser.add_argument("rout", type=int, help="number of elements in r")
parser.add_argument("rn", type=int, help="number of elements in r")
parser.add_argument("qn", type=int, help="number of elements in theta")
parser.add_argument("zn", type=int, help="number of elements in z")


def main(args: argparse.Namespace):
    mesh, quad = create_cylinder_mesh(
        cast(float, args.rin),
        cast(float, args.rout),
        cast(float, args.l),
        cast(float, args.b),
        cast(T3[int], (args.xn, args.yn, args.zn)),
        cast(Literal["x", "y", "z"], args.axis),
        cast(bool, args.make_quad),
    )
    mesh.save(args.prefix)
    quad.save(args.prefix + "_quad") if quad else ...


# ----  Here beging the main program  ---------------------------------------
# Get the command line arguments
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
