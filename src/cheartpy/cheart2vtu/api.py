from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pytools.logging.api import BLogger
from pytools.logging.trait import LOG_LEVEL, LogLevel

from .core import export_boundary, run_exports_in_parallel, run_exports_in_series
from .parser_main import get_api_args, get_cmdline_args
from .prep import init_variable_cache, parse_cmdline_args
from .print_headers import (
    print_guard,
    print_header,
    print_index_info,
)
from .trait import CmdLineArgs


def cheart2vtu(cmd_args: CmdLineArgs) -> None:
    log = BLogger(cmd_args.log)
    print_header()
    inp, indexer = parse_cmdline_args(cmd_args)
    print_index_info(indexer)
    print_guard()
    cache = init_variable_cache(inp, indexer)
    log.debug(cache)
    export_boundary(inp, cache)
    if inp.cores > 1:
        run_exports_in_parallel(inp, indexer, cache, log)
    else:
        run_exports_in_series(inp, indexer, cache, log)
    print_guard()


def cheart2vtu_api(
    prefix: str | None = None,
    mesh: str | tuple[str, str, str] = "mesh",
    space: str | None = None,
    variables: Sequence[str] = [],
    input_dir: str = "",
    output_dir: str = "",
    index: tuple[int, int, int] | None = None,
    subindex: tuple[int, int, int] | Literal["auto", "none"] | None = "none",
    time_series: str | None = None,
    binary: bool = False,
    compression: bool = True,
    progress_bar: bool = True,
    cores: int = 1,
    log: LOG_LEVEL = "INFO",
) -> None:
    args = get_api_args(
        prefix=prefix,
        index=index,
        subindex=subindex,
        vars=variables,
        input_dir=input_dir,
        output_dir=output_dir,
        mesh=mesh,
        space=space,
        time_series=time_series,
        binary=binary,
        compression=compression,
        progress_bar=progress_bar,
        cores=cores,
        log=LogLevel[log],
    )
    cheart2vtu(args)


def cheart2vtu_cli(cmd_args: Sequence[str] | None = None) -> None:
    args = get_cmdline_args(cmd_args)
    cheart2vtu(args)
