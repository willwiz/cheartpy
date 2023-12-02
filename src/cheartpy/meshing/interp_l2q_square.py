#!/usr/bin/env python3

import os
import argparse
from argparse import RawTextHelpFormatter
from cheartpy.io.io import read_array_int, write_array_int
import numpy as np
from numpy import zeros, full
from math import floor
from typing import Callable, List
from cheartpy.types import Arr, i32, f64
from cheartpy.tools.progress_bar import progress_bar
from cheartpy.io.cheartio import (
    CHRead_d_utf,
    CHRead_d_binary,
    CHRead_t_utf,
    CHWrite_d_utf,
    CHWrite_d_binary,
)


parser = argparse.ArgumentParser(
    description="""
      Create scarse array map for mapping variable from linear topologies to quadratic topologies.
      This tool has 2 modes:
        (1) making the map from Lin to Quad topology (--make-map)
        (2) mapping an array from Lin to Quad topology (default)
      --make-map
        Has 2 arguments: linear topology file name, quadratic topology file name
        Also requires 1 cmd arguments: output file name
      default
        Require 2 cmd arguments: file name of map from Lin to Quad, file name of array to be mapped
        Output file name is made by default by appending -quad to the end
        Output file name file name can also be supplied by --prefix
        --index enables batch mode assuming the files have similar names like:
          {name}-#.D
          default output file name is {name}-quad-#.D
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
    "--binary",
    action="store_true",
    help="OPTIONAL: assumes that the .D files being imported is binary",
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
    "--prefix",
    "-p",
    dest="prefix",
    type=str,
    default=None,
    help="OPTIONAL: output file will have [tag] appended to the end of name before index numbers and extension",
)
parser.add_argument(
    "--batch",
    "-b",
    dest="batch",
    action="store_true",
    help="OPTIONAL: enable looping through multiple variables",
)
parser.add_argument(
    "name",
    nargs="+",
    help="names to files. If --make-map, then last name is the file name, else first name is the map and second name is the input",
    type=str,
)


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
    rows_lin, _ = lin.shape
    rows_quad, _ = quad.shape
    if rows_lin != rows_quad:
        raise ValueError("Topologies do not have the same number of elements")
    top_map = full((quad_n, 5), -2, dtype=int)
    i = j = 0
    try:
        for i in range(rows_quad):
            for j in range(4):
                edit_val(top_map, quad[i, j], [1, lin[i, j], 0, 0, 0])
            edit_val(
                top_map,
                quad[i, 4],
                [2, top_map[quad[i, 0], 1], top_map[quad[i, 1], 1], 0, 0],
            )
            edit_val(
                top_map,
                quad[i, 5],
                [2, top_map[quad[i, 0], 1], top_map[quad[i, 2], 1], 0, 0],
            )
            edit_val(
                top_map,
                quad[i, 6],
                [
                    4,
                    top_map[quad[i, 0], 1],
                    top_map[quad[i, 1], 1],
                    top_map[quad[i, 2], 1],
                    top_map[quad[i, 3], 1],
                ],
            )
            edit_val(
                top_map,
                quad[i, 7],
                [2, top_map[quad[i, 1], 1], top_map[quad[i, 3], 1], 0, 0],
            )
            edit_val(
                top_map,
                quad[i, 8],
                [2, top_map[quad[i, 2], 1], top_map[quad[i, 3], 1], 0, 0],
            )

            if update is not None:
                update()
    except LookupError as e:
        print(f"fails on element {i + 1} node {j + 1}")
        print(e)

    return top_map


def get_qual_val(
    map: Arr[int, i32], arr: Arr[tuple[int, int], f64]
) -> float | Arr[int, f64]:
    if map[0] < 1 or map[0] > 4:
        raise AssertionError(
            f"<<<ERROR: Nodal value must be the average of 1, 2, or 4 other nodes, not {map[0]}"
        )
    kind: int = map[0] + 1
    return arr[map[1:kind]].sum() / map[0]


def lin_to_quad_arr(map: np.ndarray, arr: np.ndarray) -> np.ndarray:
    rows, _ = map.shape
    _, cols = arr.shape
    res = zeros((rows, cols))
    for i, m in enumerate(map):
        res[i] = get_qual_val(m, arr)
    return res


def make_map(args):
    _, _, lin_top = CHRead_t_utf(args.make_map[0])
    _, nquad, quad_top = CHRead_t_utf(args.make_map[1])
    # Convert to python index
    lin_top = lin_top - 1
    quad_top = quad_top - 1
    bar = progress_bar(
        f"Generating Map from {args.make_map[0]} to {args.make_map[1]}",
        max=len(quad_top),
    )
    top_map = gen_map(lin_top, quad_top, nquad, bar.next)
    write_array_int(args.name[-1], top_map)


def map_vals_i(args, l2q_map, name):
    if args.prefix is None:
        root, ext = os.path.splitext(name)
        tag = root + "-quad" + ext
    else:
        tag = args.prefix
    # print(tag)
    if args.index == None:
        if args.binary:
            _, lindata = CHRead_d_binary(name)
        else:
            _, lindata = CHRead_d_utf(name)
        quadata = lin_to_quad_arr(l2q_map, lindata)
        # print(tag)
        if args.binary:
            CHWrite_d_binary(tag, quadata)
        else:
            CHWrite_d_utf(tag, quadata)
    else:
        bar = progress_bar(
            f"Interpolating {name} to quad topology",
            max=floor((args.index[1] - args.index[0]) / args.index[2]) + 1,
        )
        for i in range(args.index[0], args.index[1] + args.index[2], args.index[2]):
            fin = name + f"-{i}.D"
            fout = tag + f"-{i}.D"
            if args.binary:
                _, lindata = CHRead_d_binary(fin)
            else:
                _, lindata = CHRead_d_utf(name)
            quadata = lin_to_quad_arr(l2q_map, lindata)
            if args.binary:
                CHWrite_d_binary(fout, quadata)
            else:
                CHWrite_d_utf(fout, quadata)
            bar.next()
    print(f"<<<  Job Complete for {name}!")


def map_vals(args):
    if len(args.name) < 2:
        raise AssertionError(
            f"<<<ERROR: normal model requires 2 or 3 arguments: map, file, [filenameout]. {len(args.name)} provided: {args.name}"
        )
    l2q_map = read_array_int(args.name[0])
    if args.batch:
        for var in args.name[1:]:
            map_vals_i(args, l2q_map, var)
    else:
        map_vals_i(args, l2q_map, args.name[1])


if __name__ == "__main__":
    args = parser.parse_args()
    if args.make_map is not None:
        make_map(args)
    else:
        map_vals(args)
