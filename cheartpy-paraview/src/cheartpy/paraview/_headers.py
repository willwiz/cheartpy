from pprint import pformat
from typing import TYPE_CHECKING

from cheartpy.io.api import fix_suffix
from cheartpy.search.trait import IIndexIterator, SearchMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .struct import CmdLineArgs


def print_guard() -> str:
    return f"\n{'#' * 100}\n"


def print_header() -> Sequence[str]:
    return (
        "#" * 100,
        "    Program for converting CHeart data to vtk unstructured grid format",
        "    This program is part of the CHeart project, which is FE solver for cardiac mechanics.",
        "    Author: Andreas Hessenthaler",
        "    Modified by: Will Zhang",
        "    Data: 12/24/2024",
        "#" * 100,
    )


def print_input_info(inp: CmdLineArgs) -> Sequence[str]:
    msg = ["The retrieving data from ", inp.input_dir]
    match inp.mesh:
        case str():
            msg = [
                *msg,
                "<<< Running Program with Mode: find",
                f"The prefix for the mesh to use is {fix_suffix(inp.mesh)}",
            ]
        case (x, t, b):
            msg = [
                *msg,
                "<<< Running Program with Mode: index",
                f"The space file to use is {x}",
                f"The topology file to use is {t}",
                f"The boundary file to use is {b}",
                "<<< The varibles to add are: ",
            ]

    match inp.index:
        case SearchMode.none:
            msg = [*msg, "No variable will be used for this run"]
        case SearchMode.auto:
            msg = [*msg, "<<< Index search model is auto"]
        case (i, j, k):
            msg = [*msg, f"<<< Time step: From {i} to {j} in steps of {k}"]
    match inp.subindex:
        case SearchMode.none:
            pass
        case SearchMode.auto:
            msg = [*msg, "<<< Automatically finding subiterations"]
        case (i, j, k):
            msg = [*msg, f"<<< Sub iterations: From {i} to {j} in steps of {k}"]
    msg = [
        *msg,
        f"<<< Output file name prefix: {inp.prefix}",
        f"<<< Output folder:           {inp.output_dir}",
        f"<<< Compress VTU:            {inp.compression}",
        f"<<< Import data as binary:   {inp.binary}",
    ]
    msg = [*msg, "<<< Variables to be added are:", pformat(inp.var, compact=True)]
    if inp.time_series is not None:
        msg = [*msg, f"<<< Adding time series from {inp.time_series}"]
    return msg


def print_index_info(indexer: IIndexIterator) -> str:
    first = _last = next(iter(indexer))
    for _last in indexer:
        pass
    return f"<<<     Time step found: From {first} to {_last}"
