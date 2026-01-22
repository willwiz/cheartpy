from typing import TYPE_CHECKING, Literal, Unpack, overload

from pytools.logging.api import BLogger

from ._arg_validation import process_cmdline_args
from ._caching import init_variable_cache
from ._headers import (
    compose_header,
    header_guard,
)
from ._parser.main_parser import get_api_args, get_cmd_args, main_parser
from .core import export_boundary, run_exports_in_parallel, run_exports_in_series

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cheartpy.paraview._parser import APIKwargs, APIKwargsFind, APIKwargsIndex

    from ._parser import SUBPARSER_MODES, CmdLineArgs

__all__ = [
    "cheart2vtu",
    "cheart2vtu_api",
    "cheart2vtu_cli",
    "get_api_args",
    "get_cmd_args",
    "main_parser",
    "process_cmdline_args",
]


def cheart2vtu(cmd_args: CmdLineArgs) -> None:
    log = BLogger(cmd_args.log)
    log.disp(*compose_header())
    inp, indexer = process_cmdline_args(cmd_args, log).unwrap()
    log.disp("", header_guard())
    cache = init_variable_cache(inp, indexer).unwrap()
    log.debug(cache)
    export_boundary(inp, cache, log)
    log.disp("", header_guard())
    log.info("<<< Processing vtus.")
    if inp.cores > 1:
        run_exports_in_parallel(inp, indexer, cache, log)
    else:
        run_exports_in_series(inp, indexer, cache, log)
    log.disp("", header_guard())


@overload
def cheart2vtu_api(cmd: Literal["find"], **kwargs: Unpack[APIKwargsFind]) -> None: ...
@overload
def cheart2vtu_api(cmd: Literal["index"], **kwargs: Unpack[APIKwargsIndex]) -> None: ...
def cheart2vtu_api(cmd: SUBPARSER_MODES, **kwargs: Unpack[APIKwargs]) -> None:
    args = get_api_args(cmd=cmd, **kwargs)
    cheart2vtu(args)


def cheart2vtu_cli(cmd_args: Sequence[str] | None = None) -> None:
    args = get_cmd_args(cmd_args)
    cheart2vtu(args)
