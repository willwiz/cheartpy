from pprint import pformat
from typing import TYPE_CHECKING

from cheartpy.io.api import fix_ch_sfx
from cheartpy.search.trait import IIndexIterator, SearchMode

if TYPE_CHECKING:
    from ._parser import CmdLineArgs


def header_guard() -> str:
    return f"\n{'#' * 100}\n"


def compose_header() -> list[str]:
    return [
        "#" * 100,
        "    Program for converting CHeart data to vtk unstructured grid format",
        "    This program is part of the CHeart project, which is FE solver for cardiac mechanics.",
        "    Author: Andreas Hessenthaler",
        "    Modified by: Will Zhang",
        "    Data: 12/24/2024",
        "#" * 100,
    ]


def print_input_info(inp: CmdLineArgs) -> list[str]:
    msg = ["The retrieving data from ", str(inp.input_dir)]
    match inp.cmd:
        case "find":
            msg = [
                *msg,
                "<<< Running Program with Mode: find",
                f"The prefix for the mesh to use is {fix_ch_sfx(inp.mesh_or_top)}",
            ]
        case "index":
            msg = [
                *msg,
                "<<< Running Program with Mode: index",
                f"The space file to use is {inp.space}",
                f"The topology file to use is {inp.mesh_or_top}",
                f"The boundary file to use is {inp.boundary}",
                "<<< The varibles to add are: ",
            ]

    match inp.index:
        case None:
            msg = [*msg, "No variable will be used for this run"]
        case SearchMode():
            msg = [*msg, "<<< Index search model is auto"]
        case (i, j, k):
            msg = [*msg, f"<<< Time step: From {i} to {j} in steps of {k}"]
    match inp.subindex:
        case None:
            pass
        case SearchMode():
            msg = [*msg, "<<< Automatically finding subiterations"]
        case (i, j, k):
            msg = [*msg, f"<<< Sub iterations: From {i} to {j} in steps of {k}"]
    msg = [
        *msg,
        f"<<< Output file name prefix: {inp.prefix}",
        f"<<< Output folder:           {inp.output_dir}",
        f"<<< Compress VTU:            {inp.compress}",
        f"<<< Import data as binary:   {inp.binary}",
    ]
    return [*msg, "<<< Variables to be added are:", pformat(inp.var, compact=True)]


def compose_index_info(indexer: IIndexIterator) -> str:
    first = _last = next(iter(indexer))
    for _last in indexer:
        pass
    return f"<<<     Time step found: From {first} to {_last}"
