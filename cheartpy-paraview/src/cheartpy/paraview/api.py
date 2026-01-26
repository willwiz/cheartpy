from typing import TYPE_CHECKING, Literal, Unpack, overload

from pytools.logging import BLogger

from ._arg_validation import process_cmdline_args
from ._caching import init_variable_cache
from ._core import export_boundary, run_exports_in_parallel, run_exports_in_series
from ._headers import compose_header, header_guard
from ._parser.main_parser import (
    get_api_args,
    get_api_args_find,
    get_api_args_index,
    get_cmd_args,
    get_vtu_cmd_args,
)
from ._parser.types import (
    SUBPARSER_MODES,
    APIKwargs,
    APIKwargsFind,
    APIKwargsIndex,
    TimeProgArgs,
    VTUProgArgs,
)
from ._time_series import (
    create_time_series,
    create_time_series_api,
    create_time_series_cli,
    create_time_series_core,
    create_time_series_json,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "cheart2vtu_api",
    "cheart2vtu_cli",
    "cheart2vtu_find",
    "cheart2vtu_index",
    "create_time_series_api",
    "create_time_series_cli",
    "create_time_series_core",
    "create_time_series_json",
]


def cheart2vtu(cmd_args: VTUProgArgs) -> None:
    log = BLogger(cmd_args.log)
    log.disp(*compose_header())
    inp, indexer = process_cmdline_args(cmd_args, log).unwrap()
    log.disp("", header_guard())
    cache = init_variable_cache(inp, indexer).unwrap()
    log.debug(cache)
    export_boundary(inp, cache.top, log)
    log.disp("", header_guard())
    if inp.mpi is None:
        log.info("<<< Processing vtus in series.")
        run_exports_in_series(inp, indexer, cache, log)
    else:
        for k, v in inp.mpi.items():
            log.info(f"<<< Processing vtus with {v} {k}.")
            break
        run_exports_in_parallel(inp.mpi, inp, indexer, cache, log)
    log.disp("", header_guard())


@overload
def cheart2vtu_api(cmd: Literal["find"], **kwargs: Unpack[APIKwargsFind]) -> None: ...
@overload
def cheart2vtu_api(cmd: Literal["index"], **kwargs: Unpack[APIKwargsIndex]) -> None: ...
def cheart2vtu_api(cmd: SUBPARSER_MODES, **kwargs: Unpack[APIKwargs]) -> None:
    args = get_api_args(cmd=cmd, **kwargs)
    cheart2vtu(args)


def cheart2vtu_find(**kwargs: Unpack[APIKwargsFind]) -> None:
    args = get_api_args_find(**kwargs)
    cheart2vtu(args)


def cheart2vtu_index(**kwargs: Unpack[APIKwargsIndex]) -> None:
    args = get_api_args_index(**kwargs)
    cheart2vtu(args)


def cheart2vtu_cli(cmd_args: Sequence[str] | None = None) -> None:
    args = get_vtu_cmd_args(cmd_args)
    cheart2vtu(args)


def main_cli(cmdline: Sequence[str] | None = None) -> None:
    args = get_cmd_args(cmdline)
    match args:
        case VTUProgArgs():
            cheart2vtu(args)
        case TimeProgArgs():
            create_time_series(args)
