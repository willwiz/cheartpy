__all__ = ["CmdLineArgs", "get_cmdline_args"]
import argparse
import dataclasses as dc
from collections.abc import Sequence
from typing import Final, Literal

################################################################################################
# The argument parse
main_parser = argparse.ArgumentParser()

################################################################################################
# The shared parser
parser = argparse.ArgumentParser(
    "time",
    description="converts cheart output Dfiles into vtu files with time steps for paraview",
)
parser.add_argument(
    "--folder",
    "-f",
    dest="folder",
    action="store",
    default="",
    type=str,
    help="supply a name for the folder to store the vtu outputs",
)
parser.add_argument(
    "prefix",
    action="store",
    type=str,
    metavar=("prefix"),
    help="supply a name for the folder to store the vtu outputs",
)
parser.add_argument(
    "time",
    type=str,
    metavar=("time"),
    help="supply a name for the folder to store the vtu outputs",
)


@dc.dataclass(slots=True)
class CmdLineArgs:
    cmd: Literal["index", "find"]
    folder: str
    prefix: Final[str]
    time_series: str


def get_cmdline_args(cmd_args: Sequence[str] | None = None) -> CmdLineArgs:
    args = main_parser.parse_args(cmd_args)
    return CmdLineArgs(
        cmd=args.cmd,
        folder=args.folder,
        prefix=args.prefix,
        time_series=args.time,
    )
