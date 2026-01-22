import argparse
import dataclasses as dc
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ["_CmdLineArgs", "get_cmdline_args"]

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
    type=Path,
    help="supply a name for the folder to store the vtu outputs",
)
time_group = parser.add_mutually_exclusive_group(required=True)
time_group.add_argument(
    "--time-step",
    dest="time",
    type=float,
    help="Time step (float). Disp-100.D would correspond to time = 100 * time_step",
)
time_group.add_argument(
    "--time-file",
    "-t",
    dest="time",
    type=str,
    help="File (Path). File containing a 1D array of floats",
)
parser.add_argument(
    "prefix",
    action="store",
    type=str,
    metavar=("prefix"),
    help="supply a name for the folder to store the vtu outputs",
)


@dc.dataclass(slots=True)
class _CmdLineArgs:
    prefix: str
    time: str | float
    root: Path


def get_cmdline_args(cmd_args: Sequence[str] | None = None) -> _CmdLineArgs:
    return main_parser.parse_args(cmd_args, namespace=_CmdLineArgs("", 1.0, Path()))
