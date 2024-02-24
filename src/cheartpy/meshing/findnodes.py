#!/usr/bin/python
# -*- coding: utf-8 -*-

# find the nodes in a file.X that satisfies the constraints given
# The inputs of this script are:
#     filein cons1 cons2 ... consn fileout

import os
from typing import Callable
import argparse
from cheartpy.io.cheartio import CHRead_d_utf, CHRead_d_bin
from cheartpy.tools.progress_bar import ProgressBar

################################################################################################
# The argument parse
parser = argparse.ArgumentParser(
    description="converts cheart output Dfiles into vtu files for paraview"
)
parser.add_argument(
    "--prefix",
    "-p",
    dest="prefix",
    action="store",
    default=None,
    type=str,
    help="OPTIONAL: supply a prefix name to be used for the exported",
)
parser.add_argument(
    "--tolerance",
    "-tol",
    dest="tol",
    action="store",
    default=4.4408920985e-15,
    type=float,
    help="OPTIONAL: give a tolerance for when two number will be considered to be the same, default is based on double precision, i.e. 4.4408920985e-15",
)
parser.add_argument(
    "--show-progress",
    action="store_true",
    dest="progress",
    help="OPTIONAL: controls whether to show a progress bar. Default is True.",
)
parser.add_argument(
    "--no-progress",
    action="store_false",
    dest="progress",
    help="OPTIONAL: controls whether to show a progress bar. Default is True.",
)
parser.add_argument(
    "--binary",
    action="store_true",
    help="OPTIONAL: assumes that the .D files being imported is binary",
)
parser.add_argument(
    "mesh",
    action="store",
    default=None,
    type=str,
    help="Mandatory: give the path to the file.",
)
parser.add_argument(
    "cons",
    nargs="+",
    action="store",
    default=None,
    type=str,
    metavar=("var"),
    help="Optional: specify the variables to add to the vtu files. Multiple variable can be listed consecutively.",
)
################################################################################################


################################################################################################


# These function compares two values and see if they are numerically equal
def comp_val(f: Callable, v, tol=4.4408920985e-15, bar=None):
    if isinstance(bar, ProgressBar):
        bar.next()
    return abs(f(*v)) < tol


def comp_ieq(f: Callable, v, bar=None):
    if isinstance(bar, ProgressBar):
        bar.next()
    return f(*v)


def main(args=None):
    args = parser.parse_args(args=args)
    if args.prefix is not None:
        name = args.prefix
    else:
        root, _ = os.path.splitext(args.mesh)
        name = root + ".nodes"
    if args.binary:
        mesh = CHRead_d_bin(args.mesh)
    else:
        mesh = CHRead_d_utf(args.mesh)
    n, dim = mesh.shape
    nodes = list(range(n))
    for s in args.cons:
        if args.progress:
            bart = ProgressBar(f"Working on {s}:", max=len(mesh))
        else:
            bart = None
        if dim == 2:
            def func(x, y): return eval(str(s))
            if isinstance(func(0.0, 0.0), (bool)):
                type = True
            elif isinstance(func(0.0, 0.0), (int, float)):
                type = False
            else:
                print(f"The constraint {s} given cannot be evaluated!!!")
                print("Please make sure the inequalities are functions of x, y")
                exit()
        elif dim == 3:
            def func3D(x, y, z): return eval(str(s))
            if isinstance(func3D(0.0, 0.0, 0.0), (bool)):
                type = True
            elif isinstance(func3D(0.0, 0.0, 0.0), (int, float)):
                type = False
            else:
                print(f"The constraint {s} given cannot be evaluated!!!")
                print("Please make sure the inequalities are functions of x, y")
                exit()
        else:
            raise ValueError("Only 2D and 3D problems supported.")
        if type:
            nodes = [p for p in nodes if comp_ieq(func3D, mesh[p], bar=bart)]
        else:
            nodes = [
                p for p in nodes if comp_val(func3D, mesh[p], tol=args.tol, bar=bart)
            ]
    with open(name, "w") as f:
        print("\nNow writing to file.")
        f.write("{}".format(len(nodes)))
        for p in nodes:
            f.write("\n{}".format(p + 1))
    print("The filter array has been written to {}".format(name))


if __name__ == "__main__":
    main()
