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


import json
import typing as tp
from pathlib import Path

from .parser_time import CmdLineArgs, get_cmdline_args
from .time_series import create_time_series_file


def xml_write_header(f: tp.TextIO) -> None:
    f.write(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1"\n'
        '         byte_order="LittleEndian"\n'
        '         compressor="vtkZLibDataCompressor">\n'
        "  <Collection>\n",
    )


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
            n = int(next(f).strip())
        except ValueError:
            print(">>>ERROR: check file format. Time series data has 1 int for header.")
            raise
        except Exception:
            raise
        for i, line in enumerate(f):
            s, v = line.strip().split()
            arr[int(s)] = float(v) + arr[i]
        if len(arr) != n:
            print()
            msg = (
                ">>>ERROR: Incorrect number of time steps in the file."
                f"          check file format. Expected {n} lines, got {len(arr)}."
            )
            raise ValueError(msg)
    return len(arr), arr


def print_cmd_header(inp: CmdLineArgs) -> None:
    print(
        "################################################################################################",
        "    script for putting together a collection with the time serie added",
        "################################################################################################\n",
        f"<<< Output folder:          {inp.folder}",
        f"<<< Input file name prefix: {inp.prefix}",
        f"<<< Output file name:       {inp.prefix + '.json'}\n",
    )


def main() -> None:
    args = get_cmdline_args()
    print_cmd_header(args)
    root = Path(args.folder)
    fout = root / (args.prefix + ".json")
    prefix = args.prefix
    time = args.time_series
    time = create_time_series_file(prefix, time, root=root)
    with fout.open("w") as f:
        json.dump(time, f)


if __name__ == "__main__":
    main()
