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
import argparse
import os.path
import typing as tp
from .tools.progress_bar import ProgressBar
from .cheart2vtu_core.time_parser import parser, InputArgs


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


def check_args(args: argparse.Namespace) -> InputArgs:
    inp = InputArgs(
        args.prefix, args.irange[0], args.irange[1], args.irange[2], args.outfolder
    )
    if args.time_series is None:
        for i in range(inp.i0, inp.it, inp.di):
            inp.time[i] = float(i)
        inp.nt = len(inp.time)
    else:
        inp.nt, inp.time = import_time_data(args.time_series)
    file_check = False
    for i in range(inp.i0, inp.it, inp.di):
        if not os.path.isfile(os.path.join(args.outfolder, f"{args.prefix}-{i}.vtu")):
            print(f">>>ERROR: {args.prefix}-{i}.vtu cannot be found")
            file_check = True
        if not i in inp.time:
            print(f">>>ERROR: step {i} is not in the time step file")
            file_check = True
    if file_check:
        raise FileNotFoundError()

    print("All files are found.")
    return inp


def print_cmd_header(inp: InputArgs):
    print(
        "################################################################################################"
    )
    print("    script for putting together a collection with the time serie added")
    print(
        "################################################################################################"
    )
    print("")
    print(
        "<<< Output folder:                                   {}".format(inp.outfolder)
    )
    print("<<< Input file name prefix:                          {}".format(inp.prefix))
    print(
        "<<< Data series:                                     From {} to {} with increment of {}".format(
            inp.i0, inp.it, inp.di
        )
    )
    print(
        "<<< Output file name:                                {}".format(
            inp.prefix + ".pvd"
        )
    )
    print("")


def main():
    args = parser.parse_args()
    inp = check_args(args)
    print_cmd_header(inp)
    bar = ProgressBar(max=inp.nt, prefix="Processing")
    fout = os.path.join(inp.outfolder, inp.prefix + ".pvd")
    with open(fout, "w") as f:
        xml_write_header(f)
        for i in range(inp.i0, inp.it, inp.di):
            xml_write_content(
                f,
                os.path.join(
                    inp.outfolder,
                    f"{
                                inp.prefix}-{i}.vtu",
                ),
                inp.time[i],
            )
            bar.next()
        xml_write_footer(f)
    bar.finish()
    print("    The process is complete!")
    print(
        "################################################################################################"
    )


if __name__ == "__main__":
    main()
