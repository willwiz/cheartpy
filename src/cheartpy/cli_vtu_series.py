# /bin/python
#
# pvpython script to convert CHeart data to vtk unstructured grid format (vtu)
#
# author: Will Zhang (willwz@gmail.com)
#
# Usage: prefix  start  end  step
# Options:
#     --time   / -t : indicate a time file to add
#     --folder / -f : indicate a folder to work from
#     --name   / -n : indicate a output filename
from __future__ import annotations

import json
import typing as tp
from pathlib import Path

from .cheart2vtu.parser_time import CmdLineArgs, get_cmdline_args
from .cheart2vtu.time_series import create_time_series_file
from .tools.path_tools import path


def xml_write_header(f: tp.TextIO) -> None:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1"\n')
    f.write('         byte_order="LittleEndian"\n')
    f.write('         compressor="vtkZLibDataCompressor">\n')
    f.write("  <Collection>\n")


def xml_write_footer(f: tp.TextIO) -> None:
    f.write("  </Collection>\n")
    f.write("</VTKFile>\n")


def xml_write_content(f: tp.TextIO, item: str, time: float) -> None:
    f.write(f'    <DataSet timestep="{time}" group="" part="0"\n')
    f.write(f'             file="{item}"/>\n')


def import_time_data(file: Path | str) -> tuple[int, dict[int, float]]:
    arr: dict[int, float] = {}
    with Path(file).open("r") as f:
        try:
            n = int(f.readline().strip())
        except ValueError:
            print(">>>ERROR: check file format. Time series data has 1 int for header.")
            raise
        except Exception:
            raise
        arr[0] = 0.0
        for i in range(n):
            s, v = f.readline().strip().split()
            arr[int(s)] = float(v) + arr[i]
    return len(arr), arr


def print_cmd_header(inp: CmdLineArgs) -> None:
    print(
        "################################################################################################",
    )
    print("    script for putting together a collection with the time serie added")
    print(
        "################################################################################################",
    )
    print()
    print(f"<<< Output folder:          {inp.folder}")
    print(f"<<< Input file name prefix: {inp.prefix}")
    print(f"<<< Output file name:       {inp.prefix + '.json'}")
    print()


def main() -> None:
    args = get_cmdline_args()
    print_cmd_header(args)
    fout = path(args.folder, args.prefix + ".json")
    prefix = path(args.folder, args.prefix)
    time = args.time_series
    time = create_time_series_file(prefix, time)
    with Path(fout).open("w") as f:
        json.dump(time, f)


if __name__ == "__main__":
    main()
