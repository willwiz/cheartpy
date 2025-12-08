import argparse
from typing import TYPE_CHECKING, Unpack

from cheartpy.search.trait import AUTO, SearchMode
from pytools.logging.trait import LogLevel
from pytools.result import Err, Ok

from .struct import APIKwargs, CmdLineArgs

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["get_api_args", "get_cmdline_args", "main_parser"]
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
    help=(
        "OPTIONAL: supply a prefix name to be used for the exported vtu files."
        'If -p is not supplied, then "paraview" will be uses, e.g., paraview-#.D'
    ),
)
parser.add_argument(
    "--folder",
    "-f",
    dest="infolder",
    action="store",
    default="",
    type=str,
    help=(
        "OPTIONAL: supply the path to the folder where the .D files are stored. "
        "If -f is not supplied, then the path is assumed to be the current folder."
    ),
)
parser.add_argument(
    "--out-folder",
    "-o",
    dest="outfolder",
    action="store",
    default="",
    type=str,
    help=(
        "OPTIONAL: supply the path to the folder where the vtu outputs should be saved to. "
        "If -f is not supplied, then the path is assumed to be the current folder."
    ),
)
parser.add_argument(
    "var",
    nargs="*",
    action="store",
    default=[],
    type=str,
    metavar=("var"),
    help=(
        "Optional: specify the variables to add to the vtu files. "
        "Multiple variable can be listed consecutively."
    ),
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
    "index",
    help="give specific indexing for all args",
    parents=[parser],
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
    help=(
        "MANDATORY: specify the start, end, and step for the range of data files. "
        "If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory."
    ),
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
    help=(
        "OPTIONAL: specify the start, end, and step for the range of data files. "
        "If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory."
    ),
)
topgroup = parser_index.add_argument_group(title="Topology")
topgroup.add_argument(
    "--space",
    "-x",
    dest="xfile",
    action="store",
    default="mesh_FE.X",
    type=str,
    help=(
        "MANDATORY: supply a relative path and file name from the current directory to the .X file"
        ", or a variable name relative to the input folder for the Space variable."
    ),
)
topgroup.add_argument(
    "--t-file",
    "-t",
    dest="tfile",
    action="store",
    default="mesh_FE.T",
    type=str,
    help=(
        "MANDATORY: supply a relative path and file name from the current directory "
        "to the topology file, the default is mesh_FE.T"
    ),
)
topgroup.add_argument(
    "--b-file",
    "-b",
    dest="bfile",
    action="store",
    default=None,
    type=str,
    help=(
        "OPTIONAL: supply a relative path and file name from the current directory "
        "to the boundary file, the default is None"
    ),
)


parser_find = subparsers.add_parser(
    "find",
    help="determine settings automatically",
    parents=[parser],
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
    dest="xfile",
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
    help=(
        "MANDATORY: specify the start, end, and step for the range of data files. "
        "If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory."
    ),
)
sub_index_fin = parser_find.add_mutually_exclusive_group()
sub_index_fin.add_argument(
    "--subindex-auto",
    action="store_const",
    dest="subindex",
    const="auto",
    help=(
        "OPTIONAL: specify the start, end, and step for the range of data files. "
        "If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory."
    ),
)
sub_index_fin.add_argument(
    "--subindex",
    nargs=3,
    dest="subindex",
    action="store",
    metavar=("start", "end", "step"),
    help=(
        "OPTIONAL: specify the start, end, and step for the range of data files. "
        "If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory."
    ),
)
parser_find.set_defaults(subindex="none")


def get_api_args(**kwargs: Unpack[APIKwargs]) -> CmdLineArgs:
    log = kwargs.get("log", LogLevel.INFO)
    mesh = kwargs.get("mesh", "mesh")
    index = kwargs.get("index")
    args = CmdLineArgs(
        mesh=mesh,
        var=kwargs.get("vars", []),
        space=kwargs.get("space"),
        prefix=kwargs.get("prefix"),
        input_dir=kwargs.get("input_dir", ""),
        output_dir=kwargs.get("output_dir", ""),
        time_series=kwargs.get("time_series"),
        progress_bar=False if (log is LogLevel.DEBUG) else kwargs.get("progress_bar", True),
        log=log,
        binary=kwargs.get("binary", False),
        compression=kwargs.get("compression", True),
        cores=kwargs.get("cores", 1),
    )
    if len(args.var) == 0:
        return args
    match index, mesh:
        case (int(), int(), int()), _:
            args.index = index
        case None, (str(), str(), str()):
            args.index = None if index is None else index
        case None, str():
            args.index = SearchMode.auto if index is None else index
    match kwargs.get("subindex", "none"):
        case "auto":
            args.subindex = AUTO
        case "none":
            args.subindex = None
        case (int(i), int(j), int(k)):
            args.subindex = (i, j, k)
    return args


def get_cmdline_args(cmd_args: Sequence[str] | None = None) -> Ok[CmdLineArgs] | Err:
    nsp = main_parser.parse_args(args=cmd_args)
    match nsp.cmd:
        case "find":
            mesh = nsp.mesh
        case "index":
            mesh = (nsp.xfile, nsp.tfile, nsp.bfile)
        case _:
            msg = "No subprogram called, cannot proceed."
            return Err(ValueError(msg))
    kwargs: APIKwargs = {
        "prefix": nsp.prefix,
        "index": nsp.index,
        "subindex": nsp.subindex,
        "vars": nsp.var,
        "input_dir": nsp.infolder,
        "output_dir": nsp.outfolder,
        "mesh": mesh,
        "space": nsp.xfile,
        "time_series": nsp.time_series,
        "binary": nsp.binary,
        "compression": nsp.compression,
        "progress_bar": nsp.progress_bar,
        "cores": nsp.cores,
        "log": LogLevel[nsp.log],
    }
    return Ok(get_api_args(**kwargs))
