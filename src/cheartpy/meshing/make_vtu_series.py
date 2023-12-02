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
import os.path
import numpy as np
import sys
import argparse
import re
from cheartpy.tools.progress_bar import progress_bar


################################################################################################
# The argument parse
parser = argparse.ArgumentParser(
    description="converts cheart output Dfiles into vtu files with time steps for paraview"
)
parser.add_argument(
    "--time",
    "-t",
    dest="timefile",
    action="store",
    default=None,
    type=str,
    help="supply a file name for the array of times at each time step",
)
parser.add_argument(
    "--time-index",
    "-s",
    nargs=3,
    dest="trange",
    action="store",
    default=None,
    type=int,
    metavar=("start", "end", "step"),
    help="specify the start end and step for the time steps of data files",
)
parser.add_argument(
    "--folder",
    "-f",
    dest="folder",
    action="store",
    default=None,
    type=str,
    help="supply a name for the folder to store the vtu outputs",
)
parser.add_argument(
    "--name",
    "-n",
    dest="name",
    action="store",
    default=None,
    type=str,
    help="supply a name for the folder to store the vtu outputs",
)
parser.add_argument(
    "--index",
    "-i",
    nargs=3,
    dest="index",
    action="store",
    default=None,
    type=int,
    metavar=("start", "end", "step"),
    help="specify the start end and step for the time steps of data files",
)
parser.add_argument(
    "prefix",
    action="store",
    type=str,
    metavar=("prefix"),
    help="supply a name for the folder to store the vtu outputs",
)


def xml_write_header(f):
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1"\n')
    f.write('         byte_order="LittleEndian"\n')
    f.write('         compressor="vtkZLibDataCompressor">\n')
    f.write("  <Collection>\n")


def xml_write_footer(f):
    f.write("  </Collection>\n")
    f.write("</VTKFile>\n")


def xml_write_content(f, item, time):
    f.write('    <DataSet timestep="{}" group="" part="0"\n'.format(time))
    f.write('             file="{}"/>\n'.format(item))


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    l.sort(key=alphanum_key)


def get_file_list(directory, name):
    dirfiles = os.listdir(directory)
    sort_nicely(dirfiles)
    return [
        i
        for i in dirfiles
        if os.path.isfile(os.path.join(directory, i)) and i.startswith(str(name))
    ]


def import_time_data(file):
    with open(file, "r") as f:
        line = f.readline().strip().split()
        n = [int(item) for item in line][0]
        arr = np.zeros(n)
        try:
            for i in range(n):
                arr[i] = float(f.readline().strip().split()[1])
        except:
            raise ValueError(
                ">>>ERROR: Header information does not match the dimensions of the import"
            )
    return n, arr


def check_args(args):
    if args.name == None:
        args.name = args.prefix
    if args.name.split(".")[-1] != "pvd":
        args.name = args.name + ".pvd"
    if not (args.index == None):
        args.i0 = args.index[0]
        args.i1 = args.index[1]
        args.it = args.index[2]
    if not (args.trange == None):
        args.t0 = args.trange[0]
        args.t1 = args.trange[1]
        args.ts = args.trange[2]
        if not (args.index == None):
            if not ((args.i1 - args.i0) / args.it == (args.t1 - args.t0) / args.ts):
                sys.exit(
                    ">>>ERROR: The data series and time series was not given the same size"
                )
    # get run path and add trailing slash if it's not already there
    dirPrefix = os.getcwd()
    args.dir = (
        os.path.join(dirPrefix, "")
        if (args.folder == None)
        else os.path.join(dirPrefix, args.folder, "")
    )


def check_file_list(files):
    for name in files:
        if not (os.path.isfile(name)):
            sys.exit("The files {} cannot be found".format(name))
    print("All files are found.")


def get_files_and_times(args):
    if args.index == None:
        files = [
            os.path.join(args.dir, i) for i in get_file_list(args.dir, args.prefix)
        ]
        check_file_list(files)
        if args.timefile == None:
            times = [i for i in range(len(files))]
        else:
            _, time_arr = import_time_data(args.timefile)
            if args.trange == None:
                if len(time_arr) < len(files):
                    sys.exit(
                        ">>>ERROR: The not enough time steps in array for the number of vtu files!!!"
                    )
                times = [time_arr[i] for i in range(len(files))]
            else:
                tindex = range(args.i0, args.i1 + 1, args.it)
                if len(time_arr) < args.t1:
                    sys.exit(
                        ">>>ERROR: The array of times is not long enough for the provided time index values!!!"
                    )
                if len(files) != len(tindex):
                    sys.exit(
                        ">>>ERROR: The index of times step provided does not have the same length as the number of vtu files!!!"
                    )
                times = [(time_arr[t - 1] if t > 0 else 0.0) for t in tindex]
    else:
        files = [
            os.path.join(args.dir, args.prefix + "-{}.vtu".format(t))
            for t in range(args.i0, args.i1 + 1, args.it)
        ]
        check_file_list(files)
        if args.timefile == None:
            times = [i for i in range(args.i0, args.i1 + 1, args.it)]
        else:
            _, time_arr = import_time_data(args.timefile)
            if args.trange == None:
                if len(time_arr) < args.i1:
                    print(len(time_arr), args.i1)
                    sys.exit(
                        ">>>ERROR: The array of times is not long enough for the provided data index values!!!"
                    )
                times = [
                    (time_arr[t - 1] if t > 0 else 0.0)
                    for t in range(args.i0, args.i1 + 1, args.it)
                ]
            else:
                if len(time_arr) < args.t1:
                    sys.exit(
                        ">>>ERROR: The array of times is not long enough for the provided time index values!!!"
                    )
                times = [
                    (time_arr[t - 1] if t > 0 else 0.0)
                    for t in range(args.t0, args.t1 + 1, args.ts)
                ]
    return files, times


def print_cmd_header(args):
    print(
        "################################################################################################"
    )
    print("    script for putting together a collection with the time serie added")
    print(
        "################################################################################################"
    )
    print("")
    print("<<< Output folder:                                   {}".format(args.dir))
    if args.index == None:
        print(
            "<<< The collection will use all files starting with: {}".format(
                args.prefix
            )
        )
    else:
        print(
            "<<< Input file name prefix:                          {}".format(
                args.prefix
            )
        )
        print(
            "<<< Data series:                                     From {} to {} with increment of {}".format(
                args.i0, args.i1, args.it
            )
        )
    if not (args.timefile == None):
        print(
            "<<< The time at each step will be added from:        {}".format(
                args.timefile
            )
        )
    if not (args.trange == None):
        print(
            "<<< The time series will be:                         From {} to {} with increment of {}".format(
                args.t0, args.t1, args.ts
            )
        )
    print("<<< Output file name:                                {}".format(args.name))
    print("")


def main():
    args = parser.parse_args()
    check_args(args)
    print_cmd_header(args)
    files, times = get_files_and_times(args)
    bar = progress_bar("Processing", max=len(files))
    with open(os.path.join(args.dir, args.name), "w") as f:
        xml_write_header(f)
        for n, t in zip(files, times):
            xml_write_content(f, n, t)
            bar.next()
        xml_write_footer(f)
    bar.finish()
    print("    The process is complete!")
    print(
        "################################################################################################"
    )


if __name__ == "__main__":
    main()
