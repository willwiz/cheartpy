from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

from pytools.logging.api import BLogger

from .arg_validation import process_cmdline_args
from .caching import init_variable_cache
from .core import export_boundary, run_exports_in_parallel, run_exports_in_series
from .parser_main import APIKwargs, get_api_args, get_cmdline_args
from .print_headers import (
    print_guard,
    print_header,
    print_index_info,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .struct import CmdLineArgs


def cheart2vtu(cmd_args: CmdLineArgs) -> None:
    log = BLogger(cmd_args.log)
    log.disp(*print_header())
    inp, indexer = process_cmdline_args(cmd_args, log)
    log.disp(print_index_info(indexer), print_guard())
    cache = init_variable_cache(inp, indexer)
    log.debug(cache)
    export_boundary(inp, cache, log)
    if inp.cores > 1:
        run_exports_in_parallel(inp, indexer, cache, log)
    else:
        run_exports_in_series(inp, indexer, cache, log)
    log.disp(print_guard())


def cheart2vtu_api(**kwargs: Unpack[APIKwargs]) -> None:
    args = get_api_args(**kwargs)
    cheart2vtu(args)


def cheart2vtu_cli(cmd_args: Sequence[str] | None = None) -> None:
    args = get_cmdline_args(cmd_args)
    cheart2vtu(args)
