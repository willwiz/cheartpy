import argparse
from pathlib import Path
from typing import Unpack, get_args

from pytools.logging import LogEnum, LogLevel

from ._types import TimeProgArgs, TimeSeriesKwargs

################################################################################################
# The argument parse
time_parser = argparse.ArgumentParser("time", add_help=False)

################################################################################################
# The shared parser
time_parser.add_argument(
    "--folder",
    "-f",
    dest="folder",
    action="store",
    default="",
    type=Path,
    help="supply a name for the folder to store the vtu outputs",
)
time_group = time_parser.add_mutually_exclusive_group(required=True)
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
time_parser.add_argument(
    "--log",
    type=LogLevel.__value__,
    choices=get_args(LogLevel.__value__),
    default="INFO",
)
time_parser.add_argument(
    "prefix",
    action="store",
    type=str,
    metavar=("prefix"),
    help="supply the name of the vtu outputs",
)


def get_api_args_time(**kwargs: Unpack[TimeSeriesKwargs]) -> TimeProgArgs:
    return TimeProgArgs(
        cmd="time",
        prefix=kwargs["prefix"],
        time=kwargs["time"],
        log=LogEnum[kwargs.get("log", "INFO")],
        folder=kwargs.get("folder", Path()),
    )
