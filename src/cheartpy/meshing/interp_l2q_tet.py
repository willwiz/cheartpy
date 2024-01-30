#!/usr/bin/env python3

import os
import argparse
from argparse import RawTextHelpFormatter
from cheartpy.io.cheartio import (
    CHRead_d,
    CHRead_header_utf,
    CHRead_t_utf,
    CHWrite_d_utf,
)
import numpy as np
from numpy import zeros, array, full
from math import floor
from typing import Callable, List
from cheartpy.tools.progress_bar import progress_bar

parser = argparse.ArgumentParser(
    description="""
      Create sparse array for mapping variable from linear topologies to quadratic topologies for tets.
      This tool has 2 modes:
        (1) making the map from Lin to Quad topology (--make-map)
        (2) mapping an array from Lin to Quad topology (default)
      --make-map
        Has 2 arguments: linear topology file name, quadratic topology file name
        Also requires 1 positional argument (taken as first name): output file name
      default
        Require 2 cmd arguments: file name of map from Lin to Quad, file name of path/varible to be mapped
        Output file name is made by default by appending -quad
        Output file name file name can also be supplied by --prefix
      --index enables batch mode assuming the files have similar names like:
        {name}-#.D
      e.g. default output file name is {name}-quad-#.D
      """,
    formatter_class=RawTextHelpFormatter,
)
parser.add_argument(
    "--make-map",
    dest="make_map",
    type=str,
    nargs=2,
    metavar=("lin_map", "quad_map"),
    default=None,
    help="OPTIONAL: this tool will be set to make the make map mode. Requires the two topology to be supplied",
)
parser.add_argument(
    "--index",
    "-i",
    nargs=3,
    dest="index",
    type=int,
    metavar=("start", "end", "step"),
    help="OPTIONAL: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.",
)
parser.add_argument(
    "--folder",
    "-f",
    type=str,
    default=None,
    help="OPTIONAL: specify a folder for the where the variables are stored. NOT YET IMPLEMENTED",
)
parser.add_argument(
    "--prefix",
    "-p",
    dest="prefix",
    type=str,
    default=None,
    help="OPTIONAL: output file will have [tag] appended to the end of name before index numbers and extension",
)
parser.add_argument(
    "name",
    nargs="+",
    help="names to files. If --make-map, then the first name is the map file name other ones are ignored, else first name is the map and second name is the input variable being interpolated",
    type=str,
)


def read_arr_int(file):
    arr = []
    with open(file, "r") as f:
        for line in f:
            arr.append([int(m) for m in line.split()])
    return array(arr)


def write_array_int(file, data):
    with open(file, "w") as outfile:
        for i in data:
            for j in i:
                outfile.write("{:12d}".format(j))
            outfile.write("\n")
    return


def edit_val(arr: np.ndarray, ind: int, val: List[int]) -> None:
    if all(a < -1 for a in arr[ind]):
        arr[ind] = val
    elif set(arr[ind]) == set(val):
        # print(f"share node index {ind} has value {arr[ind]} which matching {val}")
        pass
    else:
        print(f"index {ind} has value {arr[ind]} which does not match {val}")
        raise LookupError(
            ">>>ERROR: tried to insert index for map which does match input from prior elements"
        )


def gen_map(
    lin: np.ndarray, quad: np.ndarray, quad_n: int, update: Callable | None = None
) -> np.ndarray:
    i = j = -1
    rows_lin, _ = lin.shape
    rows_quad, _ = quad.shape
    if rows_lin == rows_quad:
        ValueError("Topologies do not have the same number of elements")
    top_map = full((quad_n, 2), -2, dtype=int)
    try:
        for i in range(rows_quad):
            for j in range(4):
                edit_val(top_map, quad[i, j], [-1, lin[i, j]])
            edit_val(
                top_map, quad[i, 4], [top_map[quad[i, 0], 1], top_map[quad[i, 1], 1]]
            )
            edit_val(
                top_map, quad[i, 5], [top_map[quad[i, 0], 1], top_map[quad[i, 2], 1]]
            )
            edit_val(
                top_map, quad[i, 6], [top_map[quad[i, 1], 1], top_map[quad[i, 2], 1]]
            )
            edit_val(
                top_map, quad[i, 7], [top_map[quad[i, 0], 1], top_map[quad[i, 3], 1]]
            )
            edit_val(
                top_map, quad[i, 8], [top_map[quad[i, 1], 1], top_map[quad[i, 3], 1]]
            )
            edit_val(
                top_map, quad[i, 9], [top_map[quad[i, 2], 1], top_map[quad[i, 3], 1]]
            )
            if update is not None:
                update()
    except LookupError as e:
        print(f"fails on element {i} node {j}")
        print(e)

    return top_map


def get_qual_val(map: np.ndarray, arr: np.ndarray) -> np.ndarray:
    if map[0] < 0:
        res = arr[map[1]]
    else:
        res = 0.5 * (arr[map[0]] + arr[map[1]])
    return res


def lin_to_quad_arr(map: np.ndarray, arr: np.ndarray) -> np.ndarray:
    rows, _ = map.shape
    _, cols = arr.shape
    res = zeros((rows, cols))
    for i, m in enumerate(map):
        res[i] = get_qual_val(m, arr)
    return res


def make_map(args):
    lin_top = CHRead_t_utf(args.make_map[0])
    quad_top = CHRead_t_utf(args.make_map[1])
    _, nnode = CHRead_header_utf(args.make_map[1])
    # Convert to python index
    lin_top = lin_top - 1
    quad_top = quad_top - 1
    print(f"Generating Map from {args.make_map[0]} to {args.make_map[1]}:")
    bar = progress_bar(">>Progress:", max=len(quad_top))
    top_map = gen_map(lin_top, quad_top, nnode, bar.next)
    write_array_int(args.name[0], top_map)


def map_vals(args):
    if len(args.name) < 2:
        raise AssertionError(
            f"<<<ERROR: normal model requires 2 or 3 arguments: map, file, [filenameout]. {len(args.name)} provided: {args.name}"
        )
    l2q_map = read_arr_int(args.name[0])

    if args.prefix is None:
        root, ext = os.path.splitext(args.name[1])
        fout = root + "-quad" + ext
    else:
        fout = args.prefix

    if args.index == None:
        lindata = CHRead_d(args.name[1])
        quadata = lin_to_quad_arr(l2q_map, lindata)
        print(fout)
        CHWrite_d_utf(fout, quadata)
    else:
        bar = progress_bar(
            f"{args.name[1]}:",
            max=floor((args.index[1] - args.index[0]) / args.index[2]) + 1,
        )
        for i in range(args.index[0], args.index[1] + args.index[2], args.index[2]):
            fin = args.name[1] + f"-{i}.D"
            name = fout + f"-{i}.D"
            lindata = CHRead_d(fin)
            quadata = lin_to_quad_arr(l2q_map, lindata)
            CHWrite_d_utf(name, quadata)
            bar.next()
    print("<<<  Job Complete!")


def main_cli(args: argparse.Namespace):
    if args.make_map is not None:
        make_map(args)
    else:
        map_vals(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main_cli(args)
