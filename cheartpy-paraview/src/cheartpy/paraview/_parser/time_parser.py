import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from ._types import TimeProgArgs

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ["get_cmdline_args"]

################################################################################################
# The argument parse
time_parser = argparse.ArgumentParser("time", add_help=False)

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
    type=Path,
    help="File (Path). File containing a 1D array of floats",
)
parser.add_argument(
    "prefix",
    action="store",
    type=str,
    metavar=("prefix"),
    help="supply the name of the vtu outputs",
)


def get_cmdline_args(cmd_args: Sequence[str] | None = None) -> TimeProgArgs:
    return time_parser.parse_args(cmd_args, namespace=TimeProgArgs("time", "", 1.0, Path()))
