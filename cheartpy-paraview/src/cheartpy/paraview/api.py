from typing import TYPE_CHECKING, Unpack

from pytools.logging.api import BLogger

from ._arg_validation import process_cmdline_args
from ._caching import init_variable_cache
from ._deprecated_parser_main import APIKwargs, get_api_args, get_cmdline_args
from ._headers import (
    print_guard,
    print_header,
    print_index_info,
)
from ._parser.main_parser import get_cmd_args, main_parser
from .core import export_boundary, run_exports_in_parallel, run_exports_in_series

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._parser import SUBPARSER_MODES, CmdLineArgs

__all__ = [
    "cheart2vtu",
    "cheart2vtu_api",
    "cheart2vtu_cli",
    "get_cmd_args",
    "main_parser",
    "process_cmdline_args",
]


def cheart2vtu(cmd_args: CmdLineArgs) -> None:
    log = BLogger(cmd_args.log)
    log.disp(*print_header())
    inp, indexer = process_cmdline_args(cmd_args, log).unwrap()
    log.disp(print_index_info(indexer), print_guard())
    cache = init_variable_cache(inp, indexer).unwrap()
    log.debug(cache)
    export_boundary(inp, cache, log)
    if inp.cores > 1:
        run_exports_in_parallel(inp, indexer, cache, log)
    else:
        run_exports_in_series(inp, indexer, cache, log)
    log.disp(print_guard())


def cheart2vtu_api(cmd: SUBPARSER_MODES, **kwargs: Unpack[APIKwargs]) -> None:
    args = get_api_args(cmd=cmd, **kwargs)
    cheart2vtu(args)


def cheart2vtu_cli(cmd_args: Sequence[str] | None = None) -> None:
    args = get_cmdline_args(cmd_args).unwrap()
    cheart2vtu(args)
