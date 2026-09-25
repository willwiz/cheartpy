from typing import TYPE_CHECKING

from cheartpy.io import fix_ch_sfx
from cheartpy.search import IIndexIterator, SearchMode
from pytools.parsing import ppfmt

if TYPE_CHECKING:
    from ._parser import VTUProgArgs

_H_LEN_ = 30


def header_guard() -> str:
    return f"{'#' * 100}"


def compose_header() -> list[str]:
    return [
        header_guard(),
        "    Program for converting CHeart data to vtk unstructured grid format",
        "    This program is part of the CHeart project, a FE solver for cardiac mechanics.",
        "    Author: Andreas Hessenthaler (Original)",
        "            Will Zhang",
        "    Date: 9/25/2026",
        header_guard(),
    ]


def format_input_info(inp: VTUProgArgs) -> list[str]:
    msg = (f"{'<<< Running Program with Mode:':<{_H_LEN_}} {inp.cmd}",)
    if inp.cmd == "find":
        msg = [*msg, f"{'<<< Assume mesh prefix is:':<{_H_LEN_}} {fix_ch_sfx(inp.top.stem)}"]
    msg = [
        *msg,
        f"{'<<< The final space file:':<{_H_LEN_}} {inp.space}",
        f"{'<<< The final topology file:':<{_H_LEN_}} {inp.top}",
        f"{'<<< The final boundary file:':<{_H_LEN_}} {inp.boundary}",
    ]

    match inp.index:
        case SearchMode.none:
            msg = [*msg, f"{'<<< No variable will be used for this run.':<{_H_LEN_}}"]
        case SearchMode.auto:
            msg = [*msg, f"{'<<< Index search model is:':<{_H_LEN_}} {'auto'}"]
        case (i, j, k):
            msg = [*msg, f"{f'<<< Time step: From {i} to {j} in steps of {k}':<{_H_LEN_}}"]
    match inp.subindex:
        case SearchMode.none: ...  # fmt: skip
        case SearchMode.auto:
            msg = [*msg, f"{'<<< Automatically finding subiterations.':<{_H_LEN_}}"]
        case (i, j, k):
            msg = [*msg, f"{'<<< Sub iterations:':<{_H_LEN_}} {f'{i} to {j} in steps of {k}'}"]
    return [
        *msg,
        f"{'<<< Output file name prefix:':<{_H_LEN_}} {inp.prefix}",
        f"{'<<< Output folder:':<{_H_LEN_}} {inp.output_dir}",
        f"{'<<< Compress VTU:':<{_H_LEN_}} {inp.compress}",
        f"{'<<< Import data as binary:':<{_H_LEN_}} {inp.binary}",
        f"{'<<< Retrieving data from:':<{_H_LEN_}} {inp.input_dir}",
        f"{'<<< Nodal variables:':<{_H_LEN_}} {ppfmt(inp.point_var, wrap_limit=100 - _H_LEN_)}",
        f"{'<<< Cell variables:':<{_H_LEN_}} {ppfmt(inp.cell_var, wrap_limit=100 - _H_LEN_)}",
    ]


def compose_index_info(indexer: IIndexIterator) -> str:
    indicies = sorted(indexer)
    return (
        f"{'<<< Time step found:':<{_H_LEN_}}"
        f" From {indicies[0]} to {indicies[-1]} in {len(indicies)} steps"
    )
