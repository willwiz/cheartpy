from cheartpy.cheart2vtu_core.data_types import CmdLineArgs
import argparse

main_parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(
    description="converts cheart output Dfiles into vtu files for paraview",
    add_help=False,
)
parser.add_argument(
    "--prefix",
    "-p",
    type=str,
    default="paraview",
    help='OPTIONAL: supply a prefix name to be used for the exported vtu files. If -p is not supplied, then "paraview" will be used. that is the outputs will be named paraview-#.D',
)
parser.add_argument(
    "--folder",
    "-f",
    dest="infolder",
    action="store",
    default="",
    type=str,
    help="OPTIONAL: supply the path to the folder where the .D files are stored. If -f is not supplied, then the path is assumed to be the current folder.",
)
parser.add_argument(
    "--out-folder",
    "-o",
    dest="outfolder",
    action="store",
    default="",
    type=str,
    help="OPTIONAL: supply the path to the folder where the vtu outputs should be saved to. If -f is not supplied, then the path is assumed to be the current folder.",
)
parser.add_argument(
    "var",
    nargs="*",
    action="store",
    default=list(),
    type=str,
    metavar=("var"),
    help="Optional: specify the variables to add to the vtu files. Multiple variable can be listed consecutively.",
)


settinggroup = parser.add_argument_group(title="Settings")
settinggroup.add_argument(
    "--no-progressbar",
    action="store_true",
    help="OPTIONAL: controls whether to show a progress bar. Default is True.",
)
settinggroup.add_argument(
    "--verbose", "-v", action="store_true", help="OPTIONAL: print more info"
)
settinggroup.add_argument(
    "--binary",
    action="store_true",
    help="OPTIONAL: assumes that the .D files being imported is binary",
)
settinggroup.add_argument(
    "--no-compression",
    dest="compression",
    action="store_false",
    help="OPTIONAL: disable compression.",
)
settinggroup.add_argument(
    "--cores",
    "-c",
    action="store",
    default=1,
    type=int,
    metavar=("#"),
    help="OPTIONAL: use multicores.",
)

extrasgroup = parser.add_argument_group(title="Extras")
extrasgroup.add_argument(
    "--make-time-series",
    dest="time_series",
    default=None,
    type=str,
    help="OPTIONAL: incorporate time data, supply a file for the time step.",
)


subparsers = main_parser.add_subparsers(help="Collective of subprogram", dest="cmd")


parser_index = subparsers.add_parser(
    "index", help="give specific indexing for all args", parents=[parser]
)
parser_index.add_argument(
    "--index",
    "-i",
    nargs=3,
    dest="index",
    action="store",
    default=[0, 0, 1],
    type=int,
    metavar=("start", "end", "step"),
    help="MANDATORY: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.",
)
parser_index.add_argument(
    "--use-subiter",
    "-si",
    dest="subiter",
    nargs=3,
    action="store",
    default=None,
    type=int,
    metavar=("start", "end", "step"),
    help="OPTIONAL: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.",
)
parser_index.add_argument(
    "--use-subauto",
    "-sa",
    dest="subauto",
    action="store_true",
    help="OPTIONAL: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.",
)
topgroup = parser_index.add_argument_group(title="Topology")
topgroup.add_argument(
    "--space",
    "-x",
    dest="xfile",
    action="store",
    default="mesh_FE.X",
    type=str,
    help='MANDATORY: supply a relative path and file name  from the current directory to the .X file (must end in .X), or a variable name relative to the input folder for the Space variable. default is "Space" in cuurent folder',
)
topgroup.add_argument(
    "--t-file",
    "-t",
    dest="tfile",
    action="store",
    default="mesh_FE.T",
    type=str,
    help="MANDATORY: supply a relative path and file name from the current directory to the topology file, the default is mesh_FE.T",
)
topgroup.add_argument(
    "--b-file",
    "-b",
    dest="bfile",
    action="store",
    default=None,
    type=str,
    help="OPTIONAL: supply a relative path and file name  from the current directory to the boundary file, the default is None",
)


parser_find = subparsers.add_parser(
    "find", help="determine settings automatically", parents=[parser]
)
parser_find.add_argument(
    "--mesh",
    dest="mesh",
    action="store",
    default="mesh",
    type=str,
    help="OPTIONAL: supply a directory which contains the mesh files",
)
parser_find.add_argument(
    "--step",
    dest="step",
    action="store",
    default=None,
    type=int,
    help="OPTIONAL: enforce step size rather than use all data found. NOT IMPLEMENTED.",
)
parser_find.add_argument(
    "--index",
    "-i",
    nargs=3,
    dest="index",
    action="store",
    default=None,
    type=int,
    metavar=("start", "end", "step"),
    help="MANDATORY: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.",
)
parser_find.add_argument(
    "--use-subiter",
    "-si",
    dest="subiter",
    nargs=3,
    action="store",
    default=None,
    type=int,
    metavar=("start", "end", "step"),
    help="OPTIONAL: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.",
)
parser_find.add_argument(
    "--use-subauto",
    "-sa",
    dest="subauto",
    action="store_true",
    help="OPTIONAL: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.",
)


def fix_suffix(prefix: str, suffix: str = "_FE.") -> str:
    for i in range(len(suffix), 0, -1):
        if prefix.endswith(suffix[:i]):
            return prefix + suffix[i:]
    return prefix + suffix


def parse_findmode_args(nsp: argparse.Namespace):
    subs: str = fix_suffix(nsp.mesh)
    space = subs + "X"
    topology = subs + "T"
    boundary = subs + "B"
    return space, topology, boundary, None


def parse_indexmode_args(nsp: argparse.Namespace):
    spacename: list[str] = nsp.xfile.split("+")
    space = spacename[0]
    disp = spacename[1] if len(spacename) == 2 else None
    return space, nsp.tfile, nsp.bfile, disp


def get_cmdline_args(
    cmd_args: list[str] | None = None,
) -> CmdLineArgs:
    nsp = main_parser.parse_args(args=cmd_args)
    match nsp.cmd:
        case "find":
            space, top, bnd, disp = parse_findmode_args(nsp)
        case "index":
            space, top, bnd, disp = parse_indexmode_args(nsp)
        case _:
            raise ValueError(f"No subprogram called, cannot proceed.")
    args = CmdLineArgs(
        nsp.cmd,
        nsp.var,
        nsp.prefix,
        nsp.infolder,
        nsp.outfolder,
        nsp.time_series,
        False if nsp.verbose else not nsp.no_progressbar,
        nsp.verbose,
        nsp.binary,
        nsp.compression,
        nsp.cores,
        space,
        top,
        bnd,
        disp,
    )
    if nsp.cmd == "find":
        args.step = nsp.step
    args.index = (
        nsp.index if nsp.index is None else (nsp.index[0], nsp.index[1], nsp.index[2])
    )
    args.sub_index = (
        nsp.subiter
        if nsp.subiter is None
        else (nsp.subiter[0], nsp.subiter[1], nsp.subiter[2])
    )
    args.sub_auto = nsp.subauto
    return args
