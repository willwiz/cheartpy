#!/usr/bin/python3
from __future__ import annotations

import argparse

from .quad_core.core import create_square_mesh

parser = argparse.ArgumentParser("square", description="Make a square")
parser.add_argument(
    "--prefix",
    "-p",
    type=str,
    default="square",
    help="Prefix for mesh.",
)
parser.add_argument(
    "--size",
    "-s",
    type=float,
    nargs=2,
    default=[1, 1],
    metavar=("Lx", "Ly"),
    help="size",
)
parser.add_argument(
    "--offset",
    type=float,
    nargs=2,
    default=[0, 0],
    metavar=("xoff", "yoff"),
    help="starting corner should be shifted by",
)
parser.add_argument("xn", type=int, help="number of elements in x")
parser.add_argument("yn", type=int, help="number of elements in y")


def main(args: argparse.Namespace) -> None:
    mesh = create_square_mesh(
        (args.xn, args.yn),
        tuple(args.size),
        tuple(args.offset),
    )
    mesh.save(args.prefix)


# ----  Here beging the main program  ---------------------------------------
# Get the command line arguments
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
