from typing import Sequence
from .tools.basiclogging import BLogger
from .cheart2vtu.parser_main import get_cmdline_args
from .cheart2vtu.print_headers import (
    print_guard,
    print_header,
    print_index_info,
)
from .cheart2vtu import (
    parse_cmdline_args,
    get_cmdline_args,
    init_variable_cache,
    export_boundary,
    run_exports_in_parallel,
    run_exports_in_series,
)


def main_cli(cmd_args: Sequence[str] | None = None) -> None:
    print_header()
    args = get_cmdline_args(cmd_args)
    LOG = BLogger(args.log)
    LOG.debug(args)
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


if __name__ == "__main__":
    main_cli()
