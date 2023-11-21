import dataclasses as dc
import argparse
from typing import Final

################################################################################################
# The argument parse
parser = argparse.ArgumentParser(
    "time",
    description="converts cheart output Dfiles into vtu files with time steps for paraview",
)
parser.add_argument(
    "--make-time-series",
    dest="time_series",
    default=None,
    type=str,
    help="OPTIONAL: incorporate time data, supply a file for the time step.",
)
parser.add_argument(
    "--index",
    "-i",
    nargs=3,
    dest="irange",
    default=[0, 0, 1],
    type=int,
    metavar=("start", "end", "step"),
    help="MANDATORY: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.",
)
parser.add_argument(
    "--folder",
    "-f",
    dest="outfolder",
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


@dc.dataclass(slots=True)
class CmdLineArgs:
    prefix: Final[str]
    time_series: str | None
    index: Final[tuple[int, int, int]]
    folder: str


@dc.dataclass
class InputArgs:
    prefix: str
    i0: int
    it: int
    di: int
    outfolder: Final[str]
    time: dict[int, float] = dc.field(default_factory=dict)
    nt: int = 0
