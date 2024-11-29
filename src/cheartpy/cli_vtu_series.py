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
from .tools.path_tools import path
from .cheart2vtu.time_series import create_time_series_file
from .cheart2vtu.parser_time import get_cmdline_args, CmdLineArgs


def xml_write_header(f: tp.TextIO):
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1"\n')
    f.write('         byte_order="LittleEndian"\n')
    f.write('         compressor="vtkZLibDataCompressor">\n')
    f.write("  <Collection>\n")


def xml_write_footer(f: tp.TextIO):
    f.write("  </Collection>\n")
    f.write("</VTKFile>\n")


def xml_write_content(f: tp.TextIO, item: str, time: float):
    f.write('    <DataSet timestep="{}" group="" part="0"\n'.format(time))
    f.write('             file="{}"/>\n'.format(item))


def import_time_data(file: str) -> tp.Tuple[int, tp.Dict[int, float]]:
    arr: dict[int, float] = dict()
    with open(file, "r") as f:
        try:
            n = int(f.readline().strip())
        except ValueError as e:
            print(">>>ERROR: check file format. Time series data has 1 int for header.")
            raise e
        except Exception as e:
            raise e
        arr[0] = 0.0
        try:
            for i in range(n):
                s, v = f.readline().strip().split()
                arr[int(s)] = float(v) + arr[i]
        except Exception as e:
            raise e

    return len(arr), arr


def print_cmd_header(inp: CmdLineArgs):
    print(
        "################################################################################################"
    )
    print("    script for putting together a collection with the time serie added")
    print(
        "################################################################################################"
    )
    print("")
    print(f"<<< Output folder:          {inp.folder}")
    print(f"<<< Input file name prefix: {inp.prefix}")
    print(f"<<< Output file name:       {inp.prefix + ".json"}")
    print("")


def main():
    args = get_cmdline_args()
    print_cmd_header(args)
    fout = path(args.folder, args.prefix + ".json")
    prefix = path(args.folder, args.prefix)
    time = args.time_series
    time = create_time_series_file(prefix, time)
    with open(fout, "w") as f:
        json.dump(time, f)


if __name__ == "__main__":
    main()
