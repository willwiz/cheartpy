from typing import Literal, Sequence

from ..tools.basiclogging import LOG_LEVEL, BLogger, LogLevel
from .core import export_boundary, run_exports_in_parallel, run_exports_in_series
from .interfaces import CmdLineArgs
from .parser_main import *
from .prep import init_variable_cache, parse_cmdline_args
from .print_headers import (
    print_guard,
    print_header,
    print_index_info,
)


def cheart2vtu(cmd_args: CmdLineArgs) -> None:
    LOG = BLogger(cmd_args.log)
    print_header()
    inp, indexer = parse_cmdline_args(cmd_args)
    print_index_info(indexer)
    print_guard()
    cache = init_variable_cache(inp, indexer)
    LOG.debug(cache)
    export_boundary(inp, cache)
    if inp.cores > 1:
        run_exports_in_parallel(inp, indexer, cache, LOG)
    else:
        run_exports_in_series(inp, indexer, cache, LOG)
    print_guard()


def cheart2vtu_api(
    prefix: str | None = None,
    mesh: str | tuple[str, str, str] = "mesh",
    space: str | None = None,
    vars: Sequence[str] = list(),
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
        vars=vars,
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
