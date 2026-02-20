from pprint import pformat
from typing import TYPE_CHECKING

from cheartpy.io.api import fix_ch_sfx
from cheartpy.search.trait import IIndexIterator, SearchMode

if TYPE_CHECKING:
    from ._parser.types import VTUProgArgs

_H_STR_LEN_ = 30


def header_guard() -> str:
    return f"{'#' * 100}"


def compose_header() -> list[str]:
    return [
        header_guard(),
        "    Program for converting CHeart data to vtk unstructured grid format",
        "    This program is part of the CHeart project, which is FE solver for cardiac mechanics.",
        "    Author: Andreas Hessenthaler (Original)",
        "            Will Zhang",
        "    Date: 1/20/2026",
        header_guard(),
    ]


def format_input_info(inp: VTUProgArgs) -> list[str]:
    msg = [f"{'<<< Retrieving data from:':<{_H_STR_LEN_}} {inp.input_dir}"]
    match inp.cmd:
        case "find":
            msg = [
                *msg,
                f"{'<<< Running Program with Mode:':<{_H_STR_LEN_}} find",
                f"{'<<< The mesh prefix is:':<{_H_STR_LEN_}} {fix_ch_sfx(inp.mesh_or_top)}",
            ]
        case "index":
            msg = [
                *msg,
                f"{'<<< Running Program with Mode:':<{_H_STR_LEN_}} index",
                f"{'<<< The space file to use is:':<{_H_STR_LEN_}} {inp.space}",
                f"{'<<< The topology file to use is:':<{_H_STR_LEN_}} {inp.mesh_or_top}",
                f"{'<<< The boundary file to use is:':<{_H_STR_LEN_}} {inp.boundary}",
                f"{'<<< The varibles to add are:':<{_H_STR_LEN_}} ",
            ]

    match inp.index:
        case None:
            msg = [*msg, f"{'<<< No variable will be used for this run.':<{_H_STR_LEN_}}"]
        case SearchMode():
            msg = [*msg, f"{'<<< Index search model is:':<{_H_STR_LEN_}} {'auto'}"]
        case (i, j, k):
            msg = [*msg, f"{f'<<< Time step: From {i} to {j} in steps of {k}':<{_H_STR_LEN_}}"]
    match inp.subindex:
        case None: ...  # fmt: skip
        case SearchMode():
            msg = [*msg, f"{'<<< Automatically finding subiterations.':<{_H_STR_LEN_}}"]
        case (i, j, k):
            msg = [
                *msg,
                f"{'<<< Sub iterations:':<{_H_STR_LEN_}} {f'From {i} to {j} in steps of {k}'}",
            ]
    msg = [
        *msg,
        f"{'<<< Output file name prefix:':<{_H_STR_LEN_}} {inp.prefix}",
        f"{'<<< Output folder:':<{_H_STR_LEN_}} {inp.output_dir}",
        f"{'<<< Compress VTU:':<{_H_STR_LEN_}} {inp.compress}",
        f"{'<<< Import data as binary:':<{_H_STR_LEN_}} {inp.binary}",
    ]
    return [
        *msg,
        f"{'<<< Variables to be added are:':<{_H_STR_LEN_}}",
        pformat(inp.var, compact=True),
    ]


def compose_index_info(indexer: IIndexIterator) -> str:
    indicies = sorted(indexer)
    return (
        f"{'<<<     Time step found:':<{_H_STR_LEN_}}"
        f" From {indicies[0]} to {indicies[-1]} in {len(indicies)} steps"
    )
