__all__ = ["main_parser", "get_api_args", "get_cmdline_args"]
import os
import argparse
from typing import Literal, Sequence
from ..io.indexing import SearchMode
from ..tools.basiclogging import LogLevel
from ..cheart_mesh import fix_suffix
from .interfaces import CmdLineArgs

main_parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(
    description="converts cheart output Dfiles into vtu files for paraview",
    add_help=False,
)
parser.add_argument(
    "--prefix",
    "-p",
    type=str,
    default=None,
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
    action="store_false",
    dest="progress_bar",
    help="OPTIONAL: controls whether to show a progress bar. Default is True.",
)
settinggroup.add_argument(
    "--log",
    type=str.upper,
    choices=["NULL", "FATAL", "ERROR", "WARN", "BRIEF", "INFO", "DEBUG"],
    default="INFO",
    help="OPTIONAL: print more info",
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
    "--subindex",
    "-si",
    dest="subindex",
    nargs=3,
    action="store",
    default=None,
    type=int,
    metavar=("start", "end", "step"),
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
    help="OPTIONAL: supply a prefix for the mesh files",
)
parser_find.add_argument(
    "--space",
    "-x",
    dest="space",
    action="store",
    default=None,
    type=str,
    help="OPTIONAL: supply a prefix for the mesh files",
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
sub_index_fin = parser_find.add_mutually_exclusive_group()
sub_index_fin.add_argument(
    "--subindex-auto",
    action="store_const",
    dest="subindex",
    const="auto",
    help="OPTIONAL: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.",
)
sub_index_fin.add_argument(
    "--subindex",
    nargs=3,
    dest="subindex",
    action="store",
    metavar=("start", "end", "step"),
    help="OPTIONAL: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.",
)
parser_find.set_defaults(subindex="none")


def parse_findmode_args(mesh: str):
    subs: str = fix_suffix(mesh)
    space = subs + "X"
    topology = subs + "T"
    boundary = subs + "B"
    boundary = boundary if os.path.isfile(boundary) else None
    return space, topology, boundary, None


def parse_indexmode_args(x: str, t: str, b: str):
    spacename: list[str] = x.split("+")
    space = spacename[0]
    disp = spacename[1] if len(spacename) == 2 else None
    return space, t, b, disp


def get_api_args(
    prefix: str | None = None,
    index: tuple[int, int, int] | None = None,
    subindex: tuple[int, int, int] | Literal["auto", "none"] | None = "none",
    vars: Sequence[str] = list(),
    input_dir: str = "",
    output_dir: str = "",
    mesh: str | tuple[str, str, str] = "mesh",
    space: str | None = None,
    time_series: str | None = None,
    binary: bool = False,
    compression: bool = True,
    progress_bar: bool = True,
    cores: int = 1,
    log: LogLevel = LogLevel.INFO,
) -> CmdLineArgs:
    args = CmdLineArgs(
        mesh,
        vars,
        space,
        prefix,
        input_dir,
        output_dir,
        time_series,
        False if (log == LogLevel.DEBUG) else progress_bar,
        log,
        binary,
        compression,
        cores,
    )
    if len(args.var) == 0:
        return args
    match mesh:
        case str(), str(), str():
            args.index = SearchMode.none if index is None else index
        case str():
            args.index = SearchMode.auto if index is None else index
    subindex = "none" if subindex is None else subindex
    args.subindex = SearchMode[subindex] if isinstance(subindex, str) else subindex
    return args


def get_cmdline_args(cmd_args: Sequence[str] | None = None) -> CmdLineArgs:
    nsp = main_parser.parse_args(args=cmd_args)
    match nsp.cmd:
        case "find":
            mesh = nsp.mesh
        case "index":
            mesh = (nsp.xfile, nsp.tfile, nsp.bfile)
        case _:
            raise ValueError(f"No subprogram called, cannot proceed.")
    return get_api_args(
        prefix=nsp.prefix,
        index=nsp.index,
        subindex=nsp.subindex,
        vars=nsp.var,
        input_dir=nsp.infolder,
        output_dir=nsp.outfolder,
        mesh=mesh,
        space=nsp.xfile,
        time_series=nsp.time_series,
        binary=nsp.binary,
        compression=nsp.compression,
        progress_bar=nsp.progress_bar,
        cores=nsp.cores,
        log=LogLevel[nsp.log],
    )
